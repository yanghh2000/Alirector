import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    set_seed,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)

from utils.prompter import CGECPrompter, Prompter
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
from accelerate import Accelerator

class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
            self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
            touch(os.path.join(args.output_dir, 'completed'))
            self.save_model(args, state, kwargs)

def train(
    # model/data params
    base_model: str = "",  
    data_path: str = "",
    eval_path: str = '',
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 16,       
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 150,
    val_set_size: int = 2000,
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
    lora_target_modules: List[str] = ['gate_proj', 'down_proj', 'up_proj'],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  
    resume_from_checkpoint: str = '',  # either training checkpoint or final adapter
    prompt_template_name: str = "llm/templates/baichuan_prompt.json",  
    logging_steps: int = 20,
    input_reverse: bool = False,
    baseline_mode = None, 
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank in [-1, 0]:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = {"": local_rank}  
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        # device_map = {"": local_rank}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    if local_rank in [-1, 0]:
        transformers.utils.logging.set_verbosity_info()
    

    prompter = CGECPrompter(prompt_template_name)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              trust_remote_code=True,
                                              use_fast=False) 
    print(tokenizer.__class__.__name__)

    if tokenizer._pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.__class__.__name__ == 'QWenTokenizer':
            tokenizer.pad_token_id = tokenizer.eos_token_id = tokenizer.eod_id
            tokenizer.bos_token = ''
            
    # model
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model_kwargs = dict(
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        model_kwargs['use_flash_attn'] = False
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        ),
        **model_kwargs,
    )
    print(model.__class__.__name__)
    print(model.dtype)
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # lora  
    if os.path.exists(resume_from_checkpoint):
        lora_config = LoraConfig.from_pretrained(resume_from_checkpoint)
        lora_config.inference_mode=False    # unfreeze lora model
    else:   
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
        print(lora_config)
    
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location=torch.device(local_rank))
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
        model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    def _tokenize_fn(prompts):
        result = tokenizer(
            prompts,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_points):
        if baseline_mode == 'src-src':
            full_prompts = [prompter.generate_align_prompt(
                    source=src,
                    predict=src,
                    target=trg,
                    bos_token=tokenizer.bos_token,
                    eos_token=tokenizer.eos_token,
                ) for src, pred, trg in zip(data_points["source"], data_points["pred"], data_points["target"])]
        elif baseline_mode == 'pred-pred':
            full_prompts = [prompter.generate_align_prompt(
                    source=pred,
                    predict=pred,
                    target=trg,
                    bos_token=tokenizer.bos_token,
                    eos_token=tokenizer.eos_token,
                ) for src, pred, trg in zip(data_points["source"], data_points["pred"], data_points["target"])]
        else:
            if not input_reverse:
                full_prompts = [prompter.generate_align_prompt(
                    source=src,
                    predict=pred,
                    target=trg,
                    bos_token=tokenizer.bos_token,
                    eos_token=tokenizer.eos_token,
                ) for src, pred, trg in zip(data_points["source"], data_points["pred"], data_points["target"])]
            else:
                full_prompts = [prompter.generate_align_prompt(
                    source=pred,        # reverse
                    predict=src,
                    target=trg,
                    bos_token=tokenizer.bos_token,
                    eos_token=tokenizer.eos_token,
                ) for src, pred, trg in zip(data_points["source"], data_points["pred"], data_points["target"])]
        tokenized_full_prompts = _tokenize_fn(full_prompts)
        
        if baseline_mode == 'src-src':
            prompt_no_outputs = [prompter.generate_align_prompt(
                    source=src,
                    predict=src,
                    bos_token=tokenizer.bos_token
                ) for src, pred in zip(data_points["source"], data_points["pred"])]
        elif baseline_mode == 'pred-pred':
            prompt_no_outputs = [prompter.generate_align_prompt(
                    source=pred,
                    predict=pred,
                    bos_token=tokenizer.bos_token
                ) for src, pred in zip(data_points["source"], data_points["pred"])]
        else:
            if not input_reverse:
                prompt_no_outputs = [prompter.generate_align_prompt(
                    source=src,
                    predict=pred,
                    bos_token=tokenizer.bos_token
                ) for src, pred in zip(data_points["source"], data_points["pred"])]
            else:
                prompt_no_outputs = [prompter.generate_align_prompt(
                    source=pred,    # reverse
                    predict=src,
                    bos_token=tokenizer.bos_token
                ) for src, pred in zip(data_points["source"], data_points["pred"])]
        tokenized_prompt_no_outputs = _tokenize_fn(prompt_no_outputs)
        
        for i in range(len(tokenized_full_prompts["labels"])):
            len_prompt_no_output = len(tokenized_prompt_no_outputs["input_ids"][i])

            tokenized_full_prompts["labels"][i] = [
                -100
            ] * len_prompt_no_output + tokenized_full_prompts["labels"][i][
                len_prompt_no_output:
            ]
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
    train_data.set_format('torch')
    if val_ds is not None:
        with accelerator.main_process_first():
            val_data = val_ds.map(
                generate_and_tokenize_prompt,
                batched=True,
                load_from_cache_file=True,
                num_proc=num_workers,
                remove_columns=column_names,
            )
        val_data.set_format('torch')
    else:
        val_data = None
        
    if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
        print('================ dataset examples ================')
        print('template length: ', len(tokenizer(prompter.template['align_prompt'])['input_ids']))
        print('max length: ', max([len(d) for d in train_data['input_ids']]))
        print(tokenizer.batch_decode(train_data['input_ids'][:2]))
        print(train_data[0])
        print(train_data[1])
    
    set_seed(seed)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
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
            eval_steps=0.05,
            save_steps=0.05,
            output_dir=output_dir,
            save_total_limit=20,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            load_best_model_at_end=True,
            tf32=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer,
                                                      pad_to_multiple_of=8,
                                                      return_tensors="pt",
                                                      padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained(os.path.join(output_dir, "best-model"))


if __name__ == "__main__":
    fire.Fire(train)