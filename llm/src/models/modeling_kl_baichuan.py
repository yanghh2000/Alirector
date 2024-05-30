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

def calculate_kl_loss(student_logits, teacher_logits):
    """
    p: (batch_size, num_classes)
    q: (batch_size, num_classes)
    """
    kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
    return kl_loss

class KLModel(nn.Module):
    """
    deprecated
    """
    def __init__(self,
                base_model_path,
                cor_lora_path=None,
                align_lora_path=None,
                kl_loss_weight: float = 0.1,
                lora_r: int = 8,
                lora_alpha: int = 16,
                lora_dropout: float = 0.05,
                lora_target_modules: List[str] = ['W_pack', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
                kl_loss_type: str = 'both',
                cor_device: int = 0,
                align_device: int = 1,
    ):
        super().__init__()
        self.kl_loss_weight = kl_loss_weight
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.kl_loss_type = kl_loss_type
        self.cor_device = cor_device
        self.align_device = align_device
        
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
        self.cor_model = self.get_train_model(base_model_path, cor_lora_path) 
        self.align_model = self.get_infer_model(base_model_path, align_lora_path) 
              
        self.freeze_align_model()
    
    def get_train_model(self, base_model_path, lora_path) -> PreTrainedModel:
        model_kwargs = dict(
            low_cpu_mem_usage=True,
            torch_dtype=self.torch_dtype,
            device_map={"": self.cor_device},
            trust_remote_code=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
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
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        if self.cor_device in [-1, 0]:
            print(lora_config)
    
        cor_model = get_peft_model(cor_model, lora_config)
        cor_model.config.use_cache = False
        
        if lora_path and os.path.exists(lora_path):
            checkpoint_name = os.path.join(lora_path, "adapter_model.bin")
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name, map_location=torch.device(self.cor_device))
                set_peft_model_state_dict(cor_model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

        if self.cor_device in [-1, 0]:
            cor_model.print_trainable_parameters()
        
        cor_model.to(self.cor_device)
        return cor_model
        
    def get_infer_model(self, base_model_path, lora_path) -> PreTrainedModel:
        model_kwargs = dict(
            low_cpu_mem_usage=True,
            torch_dtype=self.torch_dtype,
            device_map={"": self.align_device},
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
                torch_dtype=self.torch_dtype,
            )
            
        align_model.to(self.align_device)
        
        return align_model
    
    def freeze_align_model(self):
        for p in self.align_model.parameters():
            p.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        align_input_ids: torch.LongTensor = None,
        align_attention_mask: torch.LongTensor = None,
        align_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple:
                
        self.cor_model.train()
        self.align_model.eval()
        
        input_ids = input_ids.to(self.cor_device)
        attention_mask = attention_mask.to(self.cor_device)
        labels = labels.to(self.cor_device)
        
        align_input_ids = align_input_ids.to(self.align_device)
        align_attention_mask = align_attention_mask.to(self.align_device)
        align_labels = align_labels.cpu()
        
        cor_outputs = self.cor_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs,
        )
        loss = cor_outputs.loss
        cor_logits = cor_outputs.logits
        shift_cor_logits = cor_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        labels_mask = shift_labels.reshape(-1) != -100
        active_cor_logits = shift_cor_logits.view(-1, shift_cor_logits.size(-1))[labels_mask]
        
        if self.kl_loss_weight > 0:
            with torch.no_grad():
                align_outputs = self.align_model(
                    input_ids=align_input_ids,
                    labels=None,
                    attention_mask=align_attention_mask,
                    **kwargs,
                )   
                align_logits = align_outputs.logits.cpu().detach()
                shift_align_logits = align_logits[..., :-1, :].contiguous()
                shift_align_labels = align_labels[..., 1:].contiguous()
                align_labels_mask = shift_align_labels.reshape(-1) != -100
                active_align_logits = shift_align_logits.view(-1, shift_align_logits.size(-1))[align_labels_mask]
                
                active_align_logits = active_align_logits.to(self.cor_device)

            if self.kl_loss_type == 'both':
                kl_loss = (calculate_kl_loss(active_cor_logits, active_align_logits) + calculate_kl_loss(active_align_logits, active_cor_logits)) / 2
            elif self.kl_loss_type == 'forward-kl':
                kl_loss = calculate_kl_loss(active_cor_logits, active_align_logits)
            elif self.kl_loss_type == 'reverse-kl':
                kl_loss = calculate_kl_loss(active_align_logits, active_cor_logits)
                
            loss += self.kl_loss_weight * kl_loss
            
        return (loss,)