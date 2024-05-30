from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig,
)
from transformers.modeling_outputs import ModelOutput
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
from torch.nn import CrossEntropyLoss
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

import os

def get_student_model(
    base_model_path,
    lora_path,
    device,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=['W_pack', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
) -> PreTrainedModel:
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = dict(
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            device_map={"": device},
            trust_remote_code=True,
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        ),
        **model_kwargs,
    )
    
    cor_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    # lora  
    if lora_path and os.path.exists(lora_path):
        lora_config = LoraConfig.from_pretrained(lora_path)
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
    if device in [-1, 0]:
        print(lora_config)

    cor_model = get_peft_model(cor_model, lora_config)
    cor_model.config.use_cache = False
    
    if lora_path and os.path.exists(lora_path):
        checkpoint_name = os.path.join(lora_path, "adapter_model.bin")
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location=torch.device(device))
            set_peft_model_state_dict(cor_model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if device in [-1, 0]:
        cor_model.print_trainable_parameters()
    
    return cor_model

def get_teacher_model(
    base_model_path,
    lora_path,
    device,
) -> PreTrainedModel:
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = dict(
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        **model_kwargs,
    )
    model.config.use_cache = True
    
    if os.path.exists(lora_path):
        align_model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch_dtype,
        )
        
    # freeze align model
    for param in align_model.parameters():
        param.requires_grad = False
    
    return align_model

def get_two_teacher_model(
    base_model_path,
    lora_path,
    reverse_lora_path,
    device,
) -> PreTrainedModel:
    # base model
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = dict(
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        **model_kwargs,
    )
    model.config.use_cache = True
    
    # lora
    if os.path.exists(lora_path):
        model:PeftModel = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch_dtype,
            adapter_name='align'
        )
        
    # reverse
    model.load_adapter(reverse_lora_path, adapter_name='reverse_align')
        
    # freeze align model
    for param in model.parameters():
        param.requires_grad = False
    
    return model