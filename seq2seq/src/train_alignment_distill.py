from typing import Optional, Sequence, Dict
import fire
import argparse
import json
import os
import random
import sys
import nltk

import numpy as np
import torch
import transformers
from transformers import (
    BertTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BartForConditionalGeneration,
    set_seed,
    DefaultDataCollator,
    BartConfig,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer import logger
from transformers.trainer_utils import is_main_process
from datasets import load_dataset

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dataclasses import dataclass
from models.modeling_alignment_distill_bart import AlignmentDistillBART
from models.modeling_bart_dropsrc import BartForConditionalGenerationwithDropoutSrc
import torch.nn as nn
from typing import Union, List, Tuple, Dict, Optional, Any

@dataclass
class Seq2SeqDistillDataCollator:
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    max_cor_length: int = 128
    max_align_length: int = 128
    max_target_length: int = 128
    
    def __call__(self, batch):
        input_ids_list = []
        align_input_ids_list = []
        align_reverse_input_ids_list = []
        labels_list = []
        
        for b in batch:
            input_ids = b['input_ids']
            pad_cor_length = self.max_cor_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_cor_length
            
            align_input_ids = b['align_input_ids']
            pad_align_length = self.max_align_length - len(align_input_ids)
            align_input_ids = align_input_ids + [self.pad_token_id] * pad_align_length
            
            align_reverse_input_ids = b['align_reverse_input_ids']
            pad_align_reverse_length = self.max_align_length - len(align_reverse_input_ids)
            align_reverse_input_ids = align_reverse_input_ids + [self.pad_token_id] * pad_align_reverse_length
            
            labels = b['labels']
            pad_label_length = self.max_target_length - len(labels)
            labels = labels + [self.label_pad_token_id] * pad_label_length
            
            input_ids_list.append(input_ids)
            align_input_ids_list.append(align_input_ids)
            align_reverse_input_ids_list.append(align_reverse_input_ids)
            labels_list.append(labels)
            
        input_ids = torch.tensor(input_ids_list)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        align_input_ids = torch.tensor(align_input_ids_list)
        align_attention_mask = align_input_ids.ne(self.pad_token_id).long()
        align_reverse_input_ids = torch.tensor(align_reverse_input_ids_list)
        align_reverse_attention_mask = align_reverse_input_ids.ne(self.pad_token_id).long()
        labels = torch.tensor(labels_list)
        
        results = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'align_input_ids': align_input_ids,
            'align_attention_mask': align_attention_mask,
            'align_reverse_input_ids': align_reverse_input_ids,
            'align_reverse_attention_mask': align_reverse_attention_mask,
            'labels': labels
        }
        return results

class MyTrainer(Seq2SeqTrainer):
    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, "pytorch_model.bin")
        if os.path.exists(best_model_path):
            state_dict = torch.load(best_model_path, map_location="cpu")
            load_result = self.model.cor_bart.load_state_dict(state_dict, False)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["lm_loss"]
            loss = loss.mean().detach()

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None
        return (loss, logits, labels)
        
def main(
    # model/data params
    cor_bart_path: str = "",
    align_bart_path: str = "",
    reverse_align_bart_path: str = "",
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 4,
    micro_batch_size: int = 2,     
    num_train_epochs: int = 5,
    learning_rate: float = 1e-5,
    val_set_size: int = 100,
    seed: int = 42,
    warmup_ratio : float = 0.1,
    group_by_length: bool = False,
    max_cor_length: int = 128,
    max_align_length: int = 128,    # src + [sep] + pred
    max_target_length: int = 128,
    label_smoothing_factor: float = 0.0,
    kl_loss_weight: float = 0.1,
    alpha: float = 0.5,
    distill_way: str = 'average_loss',    # 'average_logits' or 'average_loss
    kl_loss_type: str = 'forward-kl',     # 'both', 'forward-kl', 'reverse-kl'
    logging_steps: int = 5,
    transformer: bool = False,
    lr_scheduler_type: str = "linear",
    optim: str = "adamw_torch",
    patience: int = 5,
    warmup_steps: int = 2000,
    adam_betas: tuple = (0.9, 0.999),
    dropout: float=0.1,
    src_dropout: float=0.2,
    from_scratch: bool=False,
    force_bart: bool = False,
):
    set_seed(seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = local_rank    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()

    tokenizer:BertTokenizer = BertTokenizer.from_pretrained(cor_bart_path)
    tokenizer.model_input_names = ["input_ids", "attention_mask", "align_input_ids", "align_attention_mask"]
    
    if force_bart:
        cor_bart = BartForConditionalGeneration.from_pretrained(cor_bart_path)
    else:
        if transformer:      # transformer
            dropout=dropout
            activation_function='relu'
            activation_dropout=0.0
            attention_dropout=0.0
            max_position_embeddings=512
            # cor_bart
            cor_bart_config = BartConfig.from_pretrained(cor_bart_path, dropout=dropout,
                                                activation_function=activation_function,
                                                activation_dropout=activation_dropout,
                                                attention_dropout=attention_dropout,
                                                max_position_embeddings=max_position_embeddings)
            if from_scratch:
                cor_bart = BartForConditionalGenerationwithDropoutSrc(config=cor_bart_config, src_dropout=src_dropout)
            else:
                cor_bart = BartForConditionalGenerationwithDropoutSrc.from_pretrained(cor_bart_path, src_dropout=src_dropout)
        else:
            dropout=dropout
            activation_function='gelu'
            activation_dropout=0.0
            attention_dropout=0.0
            # cor_bart
            cor_bart_config = BartConfig.from_pretrained(cor_bart_path, dropout=dropout,
                                                activation_function=activation_function,
                                                activation_dropout=activation_dropout,
                                                attention_dropout=attention_dropout)
            cor_bart = BartForConditionalGenerationwithDropoutSrc.from_pretrained(cor_bart_path, config=cor_bart_config, src_dropout=src_dropout)
        
    # align_bart
    align_bart = None
    align_bart_reverse = None
    if kl_loss_weight > 0:
        if force_bart:
            align_bart = BartForConditionalGeneration.from_pretrained(align_bart_path)
            align_bart_reverse = BartForConditionalGeneration.from_pretrained(reverse_align_bart_path)
        else:
            align_bart = BartForConditionalGenerationwithDropoutSrc.from_pretrained(align_bart_path, src_dropout=src_dropout)
            align_bart_reverse = BartForConditionalGenerationwithDropoutSrc.from_pretrained(reverse_align_bart_path, src_dropout=src_dropout)
    
    model = AlignmentDistillBART(
        cor_bart=cor_bart,
        align_bart=align_bart,
        align_bart_reverse=align_bart_reverse,
        kl_loss_weight=kl_loss_weight,
        alpha=alpha,
        kl_loss_type=kl_loss_type,
        distill_way=distill_way
    )
    
    def preprocess_function(batch):
        model_inputs = tokenizer(batch['source'],
                                max_length=max_cor_length,
                                padding=False,
                                truncation=True,
                                return_token_type_ids=False)

        labels = tokenizer(batch['target'],
                        max_length=max_target_length,
                        padding=False,
                        truncation=True,
                        return_token_type_ids=False)

        model_inputs["labels"] = labels["input_ids"]
        
        # align inputs
        align_inputs = [src + tokenizer.sep_token + pred for src, pred in zip(batch['source'], batch['pred'])]
        align_tokenize_outputs = tokenizer(align_inputs,
                                            max_length=max_align_length,
                                            padding=False,
                                            truncation=True,
                                            return_token_type_ids=False
                                            )
        model_inputs['align_input_ids'] = align_tokenize_outputs['input_ids']
        
        # align reverse inputs
        align_reverse_inputs = [pred + tokenizer.sep_token + src for src, pred in zip(batch['source'], batch['pred'])]
        align_reverse_tokenize_outputs = tokenizer(align_reverse_inputs,
                                            max_length=max_align_length,
                                            padding=False,
                                            truncation=True,
                                            return_token_type_ids=False
                                            )
        model_inputs['align_reverse_input_ids'] = align_reverse_tokenize_outputs['input_ids']

        return model_inputs

    # data
    data = load_dataset("json", data_files=data_path)
    datasets = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=seed
    )
    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]
    column_names = datasets["train"].column_names
    
    from accelerate import Accelerator
    accelerator = Accelerator()
    
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=column_names,
            load_from_cache_file=True,
        )
    with accelerator.main_process_first():
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=column_names,
            load_from_cache_file=True,
        )
        
    if local_rank == 0:
        print('================ dataset examples ================')
        print('max length: ', max([len(d) for d in train_dataset['input_ids']]))
        print(tokenizer.batch_decode(train_dataset['input_ids'][:2]))
        print(tokenizer.batch_decode(train_dataset['labels'][:2]))
        print(train_dataset[0])

    # eval_steps = 1 / num_train_epochs if not eval_steps else eval_steps
    # warmup_ratio = 1 / num_train_epochs
    adam_beta1, adam_beta2 = adam_betas
    trainer = MyTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            # warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            # bf16=True if torch.cuda.is_bf16_supported() else False,
            # fp16=False if torch.cuda.is_bf16_supported() else True,
            fp16=True,
            logging_steps=logging_steps,
            lr_scheduler_type=lr_scheduler_type,
            optim=optim,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=0.05,
            save_steps=0.05,
            output_dir=output_dir,
            save_total_limit=3,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            label_smoothing_factor=label_smoothing_factor,
            load_best_model_at_end=True,
            tf32=True,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=Seq2SeqDistillDataCollator(pad_token_id=tokenizer.pad_token_id,
                                     label_pad_token_id=-100,
                                     max_cor_length=max_cor_length,
                                     max_align_length=max_align_length,
                                     max_target_length=max_target_length),
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=patience)]
    )

    # Training
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best-model'))  # Saves the tokenizer too for easy upload
    
if __name__ == '__main__':
    fire.Fire(main)