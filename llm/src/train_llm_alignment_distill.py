import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from typing import Any, List
import fire
import torch
import transformers
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List

from transformers import (
    set_seed,
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
)

from utils.prompter import CGECPrompter
from accelerate import Accelerator
from utils.kl_baichuan_utils import *
from threading import Thread

teacher_returns = [None]
reverse_teacher_returns = [None]

def calculate_kl_loss(student_logits, teacher_logits):
    """
    p: (batch_size, num_classes)
    q: (batch_size, num_classes)
    """
    kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
    return kl_loss

def teacher_task(model: PeftModel):
    model.set_adapter('align')
    with torch.no_grad():
        teacher_outputs = model(
            input_ids=align_input_ids,
            labels=None,
            attention_mask=align_attention_mask,
        )
        teacher_logits = teacher_outputs.logits
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_teacher_labels = align_labels[..., 1:].contiguous()
        teahcer_labels_mask = shift_teacher_labels.reshape(-1) != -100
        teacher_returns[0] = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))[teahcer_labels_mask]
        
    model.set_adapter('reverse_align')
    with torch.no_grad():
        reverse_teacher_outputs = model(
            input_ids=reverse_align_input_ids,
            labels=None,
            attention_mask=align_attention_mask,
        )
        teacher_logits = reverse_teacher_outputs.logits
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_teacher_labels = align_labels[..., 1:].contiguous()
        teahcer_labels_mask = shift_teacher_labels.reshape(-1) != -100
        reverse_teacher_returns[0] = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))[teahcer_labels_mask]

class DistillTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model: PreTrainedModel = None,
        kl_loss_weight: float = 0.5,
        kl_loss_type: str = 'forward-kl',
        alpha_foward: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_loss_weight = kl_loss_weight
        self.kl_loss_type = kl_loss_type
        assert kl_loss_type in ['both', 'forward-kl', 'reverse-kl'], 'invalid kl_loss_type'
        self.alpha_foward = alpha_foward
        
    
    def _prepare_input(self, data):
        return {
            'input_ids': data['input_ids'].to(self.model.device),
            'attention_mask': data['attention_mask'].to(self.model.device),
            'align_input_ids': data['align_input_ids'].to(self.teacher_model.device),
            'align_attention_mask': data['align_attention_mask'].to(self.teacher_model.device),
            'labels': data['labels'].to(self.model.device),
            'align_labels': data['align_labels'].to(self.teacher_model.device),
            'reverse_align_input_ids': data['reverse_align_input_ids'].to(self.teacher_model.device),
            # 'reverse_align_attention_mask': data['reverse_align_attention_mask'].to(self.teacher_model.device),
            # 'reverse_align_labels': data['reverse_align_labels'].to(self.teacher_model.device),
        }
    
    def compute_loss(self, model, inputs, return_outputs=False):
        global align_input_ids, align_attention_mask, align_labels, reverse_align_input_ids
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        align_input_ids = inputs['align_input_ids']
        reverse_align_input_ids = inputs['reverse_align_input_ids']
        align_attention_mask = inputs['align_attention_mask']
        # reverse_align_attention_mask = inputs['reverse_align_attention_mask']
        labels = inputs['labels']
        align_labels = inputs['align_labels']
        # reverse_align_labels = inputs['reverse_align_labels']
        
        # teacher lm loss
        if self.kl_loss_weight > 0:
            self.teacher_model.eval()
            t1 = Thread(target=teacher_task, args=(self.teacher_model,))
            t1.start()      # student forward and teacher forward in parallel
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            t1.join()       # wait for teacher
            # student lm loss
            loss = outputs.loss
            student_logits = outputs.logits
            shift_student_logits = student_logits[..., :-1, :].contiguous()
            shift_student_labels = labels[..., 1:].contiguous()
            student_labels_mask = shift_student_labels.reshape(-1) != -100
            active_student_logits = shift_student_logits.view(-1, shift_student_logits.size(-1))[student_labels_mask]
            
            active_teacher_logits = teacher_returns[0] 
            active_reverse_teacher_logits = reverse_teacher_returns[0]           
            # ignore the loss if the length is not aligned
            if active_student_logits.size(0) != active_teacher_logits.size(0):
                return (loss, outputs) if return_outputs else loss
                
            active_teacher_logits = active_teacher_logits.to(model.device)
            active_reverse_teacher_logits = active_reverse_teacher_logits.to(model.device)
            
            if self.kl_loss_type == 'both':
                align_kl_loss = (calculate_kl_loss(active_student_logits, active_teacher_logits) + calculate_kl_loss(active_teacher_logits, active_student_logits)) / 2
                reverse_align_kl_loss = (calculate_kl_loss(active_student_logits, active_reverse_teacher_logits) + calculate_kl_loss(active_reverse_teacher_logits, active_student_logits)) / 2
                kl_loss = self.alpha_foward * align_kl_loss + (1 - self.alpha_foward) * reverse_align_kl_loss
            elif self.kl_loss_type == 'forward-kl':
                align_kl_loss = calculate_kl_loss(active_student_logits, active_teacher_logits)
                reverse_align_kl_loss = calculate_kl_loss(active_student_logits, active_reverse_teacher_logits)
                kl_loss = self.alpha_foward * align_kl_loss + (1 - self.alpha_foward) * reverse_align_kl_loss
            elif self.kl_loss_type == 'reverse-kl':
                align_kl_loss = calculate_kl_loss(active_teacher_logits, active_student_logits)
                reverse_align_kl_loss = calculate_kl_loss(active_reverse_teacher_logits, active_student_logits)
                kl_loss = self.alpha_foward * align_kl_loss + (1 - self.alpha_foward) * reverse_align_kl_loss
            
            loss += self.kl_loss_weight * kl_loss
        else:
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            # student lm loss
            loss = outputs.loss  
        
        return (loss, outputs) if return_outputs else loss

@dataclass
class LLMDistillDataCollator:
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    cutoff_len: int = 150
    
    def __call__(self, batch):
        input_ids_list = []
        align_input_ids_list = []
        reverse_align_input_ids_list = []
        align_labels_list = []
        reverse_align_labels_list = []
        labels_list = []
        
        for b in batch:
            input_ids = b['input_ids'][:self.cutoff_len]
            pad_cor_length = self.cutoff_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_cor_length
            
            align_input_ids = b['align_input_ids'][:self.cutoff_len]
            pad_align_length = self.cutoff_len - len(align_input_ids)
            align_input_ids = align_input_ids + [self.pad_token_id] * pad_align_length
            
            reverse_align_input_ids = b['reverse_align_input_ids'][:self.cutoff_len]
            reverse_pad_align_length = self.cutoff_len - len(reverse_align_input_ids)
            reverse_align_input_ids = reverse_align_input_ids + [self.pad_token_id] * reverse_pad_align_length
            
            labels = b['labels'][:self.cutoff_len]
            pad_label_length = self.cutoff_len - len(labels)
            labels = labels + [self.label_pad_token_id] * pad_label_length
            
            align_labels = b['align_labels'][:self.cutoff_len]
            pad_align_label_length = self.cutoff_len - len(align_labels)
            align_labels = align_labels + [self.label_pad_token_id] * pad_align_label_length
            
            # reverse_align_labels = b['reverse_align_labels'][:self.cutoff_len]
            # reverse_pad_align_label_length = self.cutoff_len - len(reverse_align_labels)
            # reverse_align_labels = reverse_align_labels + [self.label_pad_token_id] * reverse_pad_align_label_length
            
            input_ids_list.append(input_ids)
            align_input_ids_list.append(align_input_ids)
            labels_list.append(labels)
            align_labels_list.append(align_labels)
            reverse_align_input_ids_list.append(reverse_align_input_ids)
            # reverse_align_labels_list.append(reverse_align_labels)
            
        input_ids = torch.tensor(input_ids_list)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        align_input_ids = torch.tensor(align_input_ids_list)
        align_attention_mask = align_input_ids.ne(self.pad_token_id).long()
        labels = torch.tensor(labels_list)
        align_labels = torch.tensor(align_labels_list)
        reverse_align_input_ids = torch.tensor(reverse_align_input_ids_list)
        # reverse_align_attention_mask = reverse_align_input_ids.ne(self.pad_token_id).long()
        # reverse_align_labels = torch.tensor(reverse_align_labels_list)
        
        results = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'align_input_ids': align_input_ids,
            'align_attention_mask': align_attention_mask,
            'labels': labels,
            'align_labels': align_labels,
            'reverse_align_input_ids': reverse_align_input_ids,
            # 'reverse_align_attention_mask': reverse_align_attention_mask,
            # 'reverse_align_labels': reverse_align_labels
        }
        return results

def train(
    # model/data params
    base_model: str = '',  
    cor_lora_path = '',         # lora model path of correction model
    align_lora_path = '',       # lora model path of alignment model
    reverse_align_lora_path = '',    # lora model path of reverse alignment model
    kl_loss_weight: float = 0.5,
    kl_loss_type: str = 'forward-kl',
    data_path: str = "",
    eval_path: str = '',
    output_dir: str = "",
    eval_steps: float = 0.05,
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 16,       
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 1000,
    seed: int = 42,
    warmup_ratio: float = 0.1,
    optim='adamw_torch',
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # baichuan: ['W_pack', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    # llama: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    # Qwen: ["w1", "w2", "c_proj", "c_attn"]
    lora_target_modules: List[str] = ['W_pack', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  
    resume_from_checkpoint: str = '',  
    prompt_template_name: str = "llm/templates/baichuan_prompt.json",  
    logging_steps: int = 20,
    early_stopping_patience: int = 3,
    input_reverse: bool = False,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    if local_rank in [-1, 0]:
        transformers.utils.logging.set_verbosity_info()

    prompter = CGECPrompter(prompt_template_name)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              trust_remote_code=True,
                                              use_fast=False) 
    if tokenizer._pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
            
    # model
    student_device = local_rank
    teacher_device = local_rank + torch.cuda.device_count() // 2
    
    student_model = get_student_model(
        base_model,
        cor_lora_path,
        student_device,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target_modules
    )
    if kl_loss_weight > 0:
        teacher_model = get_two_teacher_model(base_model, align_lora_path, reverse_align_lora_path, teacher_device)
    else:
        teacher_model  = reverse_teacher_model = student_model     

    def _tokenize_fn(prompts):
        result = tokenizer(
            prompts,
            # truncation=True,
            # max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_points):
        # cor inputs
        full_prompts = [prompter.generate_prompt(
            source=src,
            target=trg,
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
        ) for src, trg in zip(data_points["source"], data_points["target"])]
        tokenized_full_prompts = _tokenize_fn(full_prompts)
        
        prompt_no_outputs = [prompter.generate_prompt(
            source=src,
            bos_token=tokenizer.bos_token
        ) for src in data_points["source"]]
        tokenized_prompt_no_outputs = _tokenize_fn(prompt_no_outputs)
        
        # align inputs
        align_full_prompts = [prompter.generate_align_prompt(
            source=src,
            predict=pred,
            target=trg,
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
        ) for src, pred, trg in zip(data_points["source"], data_points["pred"], data_points["target"])]
        reverse_align_full_prompts = [prompter.generate_align_prompt(
            source=pred,        # reverse
            predict=src,
            target=trg,
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
        ) for src, pred, trg in zip(data_points["source"], data_points["pred"], data_points["target"])]
        align_tokenized_full_prompts = _tokenize_fn(align_full_prompts)
        reverse_align_tokenized_full_prompts = _tokenize_fn(reverse_align_full_prompts)
        
        align_prompt_no_outputs = [prompter.generate_align_prompt(
            source=src,     
            predict=pred,
            bos_token=tokenizer.bos_token
        ) for src, pred in zip(data_points["source"], data_points["pred"])]
        reverse_align_prompt_no_outputs = [prompter.generate_align_prompt(
            source=pred,        # reverse
            predict=src,
            bos_token=tokenizer.bos_token
        ) for src, pred in zip(data_points["source"], data_points["pred"])]
        align_tokenized_prompt_no_outputs = _tokenize_fn(align_prompt_no_outputs)
        reverse_align_tokenized_prompt_no_outputs = _tokenize_fn(reverse_align_prompt_no_outputs)
        
        for i in range(len(align_tokenized_full_prompts["labels"])):
            # align labels 
            align_len_prompt_no_output = len(align_tokenized_prompt_no_outputs["input_ids"][i])

            align_tokenized_full_prompts["labels"][i] = [
                -100
            ] * align_len_prompt_no_output + align_tokenized_full_prompts["labels"][i][
                align_len_prompt_no_output:
            ]
            
            # reverse align labels
            reverse_align_len_prompt_no_output = len(reverse_align_tokenized_prompt_no_outputs["input_ids"][i])
            
            reverse_align_tokenized_full_prompts["labels"][i] = [
                -100
            ] * reverse_align_len_prompt_no_output + reverse_align_tokenized_full_prompts["labels"][i][
                reverse_align_len_prompt_no_output:
            ]
            
            # cor labels 
            len_prompt_no_output = len(tokenized_prompt_no_outputs["input_ids"][i])

            tokenized_full_prompts["labels"][i] = [
                -100
            ] * len_prompt_no_output + tokenized_full_prompts["labels"][i][
                len_prompt_no_output:
            ]
            # diff_len = align_len_prompt_no_output - len_prompt_no_output
            # assert diff_len > 0
            # tokenized_full_prompts["input_ids"][i] = [tokenizer.pad_token_id] * diff_len + tokenized_full_prompts["input_ids"][i]
            # tokenized_full_prompts["labels"][i] = [-100] * diff_len + tokenized_full_prompts["labels"][i]
        
        tokenized_full_prompts['align_input_ids'] = align_tokenized_full_prompts['input_ids']
        tokenized_full_prompts['align_labels'] = align_tokenized_full_prompts['labels']
        tokenized_full_prompts['reverse_align_input_ids'] = reverse_align_tokenized_full_prompts['input_ids']
        # tokenized_full_prompts['reverse_align_labels'] = reverse_align_tokenized_full_prompts['labels']
        
        return tokenized_full_prompts

    # prepocess data
    data = load_dataset("json", data_files=data_path)
    column_names = data['train'].column_names
    num_workers = 1 if tokenizer.__class__.__name__ == 'QWenTokenizer' else os.cpu_count()

    accelerator = Accelerator()
    if os.path.exists(eval_path):
        train_ds = data["train"]
        val_ds = load_dataset("json", data_files=eval_path)["train"]
        if val_set_size > 0:
            val_ds = val_ds.select(range(val_set_size))
    elif val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_ds = train_val["train"]
        val_ds = train_val["test"]
    else:
        train_ds = data["train"]
        val_ds = None
        
        
    with accelerator.main_process_first():
        train_data = train_ds.map(
            generate_and_tokenize_prompt,
            batched=True,
            num_proc=num_workers,
            load_from_cache_file=True,
            remove_columns=column_names,
        )
    if val_ds is not None:
        with accelerator.main_process_first():
            val_data = val_ds.map(
                generate_and_tokenize_prompt,
                batched=True,
                load_from_cache_file=True,
                num_proc=num_workers,
                remove_columns=column_names,
            )
    else:
        val_data = None
        
    if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
        print('================ dataset examples ================')
        print('template length: ', len(tokenizer(prompter.template['prompt_input'])['input_ids']))
        print('max length: ', max([len(d) for d in train_data['input_ids']]))
        print(tokenizer.batch_decode(train_data['input_ids'][:2]))
        print(train_data[0])
        print(train_data[1])
    
    set_seed(seed)
    
    training_args = TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=False if torch.cuda.is_bf16_supported() else True,
            logging_steps=logging_steps,
            lr_scheduler_type='cosine',     # linear, constant, cosine
            optim=optim,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=20,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            load_best_model_at_end=True,
            remove_unused_columns=False,
            tf32=True,
        )

    trainer = DistillTrainer(
        model=student_model,
        teacher_model=teacher_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=LLMDistillDataCollator(pad_token_id=tokenizer.pad_token_id,
                                     label_pad_token_id=-100,
                                     cutoff_len=cutoff_len),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        kl_loss_weight=kl_loss_weight,
        kl_loss_type=kl_loss_type,
    )

    trainer.train(resume_from_checkpoint=False)

    student_model.save_pretrained(os.path.join(output_dir, "best-model"))


if __name__ == "__main__":
    fire.Fire(train)

