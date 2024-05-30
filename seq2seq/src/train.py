from typing import Dict, List
import fire
import argparse
import json
import os
import random
import sys

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
    BartConfig,
)
from transformers.trainer_utils import is_main_process
from datasets import load_dataset
from accelerate import Accelerator
from models.modeling_copy import BartForConditionalGenerationWithCopyMech
from models.modeling_bart_dropsrc import BartForConditionalGenerationwithDropoutSrc

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def main(
    # model/data params
    model_path: str = 'fnlp/bart-large-chinese',
    data_path: str = "",
    eval_path: str = "",
    output_dir: str = "",
    val_set_size: int = 2000,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,     
    eval_batch_size: int = 32,  
    num_train_epochs: int = 5,
    learning_rate: float = 1e-5,
    seed: int = 42,
    warmup_ratio : float = 0.1,
    group_by_length: bool = False,
    max_source_length: int = 128,
    max_target_length: int = 128,
    eval_max_source_length: int = 256,
    eval_max_target_length: int = 256,
    label_smoothing_factor: float = 0.0,
    logging_steps: int = 20,
    transformer: bool = False,      # whether to use transformer or bart
    copy: bool = False,
    lr_scheduler_type: str = "linear",
    optim: str = "adamw_torch",
    patience: int = 5,
    warmup_steps: int = 2000,
    adam_betas: tuple = (0.9, 0.999),
    dropout: float=0.0,
    src_dropout: float=0.0,
    pretrained: bool=True,   # whether to load the model from pretrained model or random initialized
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

    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    model_cls = BartForConditionalGenerationWithCopyMech if copy else BartForConditionalGenerationwithDropoutSrc
    
    if transformer:      # transformer
        dropout=dropout
        activation_function='relu'
        activation_dropout=0.0
        attention_dropout=0.0
        src_dropout=src_dropout
        max_position_embeddings=512
        config = BartConfig.from_pretrained(model_path, dropout=dropout,
                                            activation_function=activation_function,
                                            activation_dropout=activation_dropout,
                                            attention_dropout=attention_dropout,
                                            max_position_embeddings=max_position_embeddings)
        if not pretrained:
            model = model_cls(config=config, src_dropout=src_dropout)
        else:
            model = model_cls.from_pretrained(model_path, src_dropout=src_dropout)
    else:
        dropout=dropout
        activation_function='gelu'
        activation_dropout=0.0
        attention_dropout=0.0
        src_dropout=src_dropout
        config = BartConfig.from_pretrained(model_path, dropout=dropout,
                                            activation_function=activation_function,
                                            activation_dropout=activation_dropout,
                                            attention_dropout=attention_dropout)
        model = model_cls.from_pretrained(model_path, config=config, src_dropout=src_dropout)
    # model.config.max_length=max_target_length

    def preprocess_function(batch: Dict[str, List], src_max_length, tgt_max_length):
        inputs = batch['source']
        targets = batch['target']
        model_inputs = tokenizer(inputs,
                                max_length=src_max_length,
                                padding=False,
                                truncation=True,
                                return_token_type_ids=False)

        # Setup the tokenizer for targets
        labels = tokenizer(targets,
                        max_length=tgt_max_length,
                        padding=False,
                        truncation=True,
                        return_token_type_ids=False)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # prepocess data
    data = load_dataset("json", data_files=data_path)
    column_names = data['train'].column_names
    num_workers = 1 if tokenizer.__class__.__name__ == 'QWenTokenizer' else os.cpu_count()

    accelerator = Accelerator()
    if os.path.exists(eval_path):
        train_ds = data["train"]
        val_ds = load_dataset("json", data_files=eval_path)["train"]
    elif val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_ds = train_val["train"]
        val_ds = train_val["test"]
    else:
        train_ds = data["train"]
        val_ds = None
        
        
    with accelerator.main_process_first():      # first load the dataset in the main process, then load the cache in other processes
        train_data = train_ds.map(
            preprocess_function,
            batched=True,
            num_proc=num_workers,
            load_from_cache_file=True,
            remove_columns=column_names,
            fn_kwargs={'src_max_length': max_source_length, 'tgt_max_length': max_target_length}
        )
    train_data.set_format('torch')
    if val_ds is not None:
        with accelerator.main_process_first():
            val_data = val_ds.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                num_proc=4,
                remove_columns=column_names,
                fn_kwargs={'src_max_length': eval_max_source_length, 'tgt_max_length': eval_max_target_length}
            )
        val_data.set_format('torch')
    else:
        val_data = None
        
    if local_rank == 0:
        print('================ dataset examples ================')
        print('max length: ', max([len(d) for d in train_data['input_ids']]))
        print(tokenizer.batch_decode(train_data['input_ids'][:2]))
        print(tokenizer.batch_decode(train_data['labels'][:2]))
        print(train_data[0])
        print(train_data[1])

    eval_steps = 1 / num_train_epochs
    warmup_ratio = 1 / num_train_epochs
    adam_beta1, adam_beta2 = adam_betas
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
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
            eval_steps=eval_steps,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=10,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            label_smoothing_factor=label_smoothing_factor,
            load_best_model_at_end=True,
            tf32=True,
        ),
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,  
        ),
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=patience)]
    )

    # Training
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best-model'))  # Saves the tokenizer too for easy upload

if __name__ == '__main__':
    fire.Fire(main)