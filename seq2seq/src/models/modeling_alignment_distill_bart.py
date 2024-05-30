from transformers import BartForConditionalGeneration, BartPretrainedModel, BartConfig, BertTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
from torch.nn import CrossEntropyLoss

import os

def calculate_kl_loss(student_logits, teacher_logits):
    """
    p: (batch_size, num_classes)
    q: (batch_size, num_classes)
    """
    kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
    return kl_loss

class AlignmentDistillBART(nn.Module):
    def __init__(
        self,
        cor_bart = BartPretrainedModel,
        align_bart: Optional[BartPretrainedModel] = None,
        align_bart_reverse: Optional[BartPretrainedModel] = None,
        kl_loss_weight: float = 0.1,    # beta
        kl_loss_type: str = 'both',
        distill_way: str = 'average_loss',    # 'average_logits' or 'average_loss'
        alpha: float = 0.5,
    ):
        super().__init__()
        self.cor_bart = cor_bart
        self.align_bart = align_bart
        self.align_bart_reverse = align_bart_reverse
        self.kl_loss_weight = kl_loss_weight
        assert kl_loss_type in ['both', 'forward-kl', 'reverse-kl'], 'invalid kl_loss_type'
        self.kl_loss_type = kl_loss_type
        self.distill_way = distill_way
        self.alpha = alpha
        
        if self.align_bart and self.align_bart_reverse:
            self.freeze_align_bart()
        
    def freeze_align_bart(self):
        for p in self.align_bart.parameters():
            p.requires_grad = False
            
        for p in self.align_bart_reverse.parameters():
            p.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        align_input_ids: torch.LongTensor = None,
        align_attention_mask: Optional[torch.Tensor] = None,
        align_reverse_input_ids: torch.LongTensor = None,
        align_reverse_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        self.cor_bart.train()
        
        cor_outputs = self.cor_bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        lm_loss = cor_outputs.loss.clone().detach()
        loss = cor_outputs.loss
        cor_logits = cor_outputs.logits
        labels_mask = labels.reshape(-1) != -100
        active_cor_logits = cor_logits.view(-1, cor_logits.size(-1))[labels_mask]
        
        if self.kl_loss_weight > 0 and align_input_ids is not None and self.align_bart is not None:
            self.align_bart.eval()
            self.align_bart_reverse.eval()
            with torch.no_grad():
                align_outputs = self.align_bart(
                    input_ids=align_input_ids,
                    attention_mask=align_attention_mask,
                    labels=labels,
                    **kwargs,
                )   
                align_logits = align_outputs.logits
                active_align_logits = align_logits.view(-1, align_logits.size(-1))[labels_mask]
                
            with torch.no_grad():
                align_reverse_outputs = self.align_bart_reverse(
                    input_ids=align_reverse_input_ids,
                    attention_mask=align_reverse_attention_mask,
                    labels=labels,
                    **kwargs,
                )   
                align_reverse_logits = align_reverse_outputs.logits
                active_align_reverse_logits = align_reverse_logits.view(-1, align_reverse_logits.size(-1))[labels_mask]
                
            if self.distill_way == 'average_logits':
                teacher_logits = self.alpha * active_align_logits + (1 - self.alpha) * active_align_reverse_logits
                
            if self.kl_loss_type == 'both':
                if self.distill_way == 'average_logits':
                    kl_loss = (calculate_kl_loss(active_cor_logits, teacher_logits) + calculate_kl_loss(teacher_logits, active_cor_logits)) / 2
                else:
                    align_kl_loss = (calculate_kl_loss(active_cor_logits, active_align_logits) + calculate_kl_loss(active_align_logits, active_cor_logits)) / 2
                    align_reverse_kl_loss = (calculate_kl_loss(active_cor_logits, active_align_reverse_logits) + calculate_kl_loss(active_align_reverse_logits, active_cor_logits)) / 2
                    kl_loss = self.alpha * align_kl_loss + (1 - self.alpha) * align_reverse_kl_loss
            elif self.kl_loss_type == 'forward-kl':
                if self.distill_way == 'average_logits':
                    kl_loss = calculate_kl_loss(active_cor_logits, teacher_logits)
                else:
                    align_kl_loss = calculate_kl_loss(active_cor_logits, active_align_logits)
                    align_reverse_kl_loss = calculate_kl_loss(active_cor_logits, active_align_reverse_logits)
                    kl_loss = self.alpha * align_kl_loss + (1 - self.alpha) * align_reverse_kl_loss
            elif self.kl_loss_type == 'reverse-kl':
                if self.distill_way == 'average_logits':
                    kl_loss = calculate_kl_loss(teacher_logits, active_cor_logits)
                else:
                    align_kl_loss = calculate_kl_loss(active_align_logits, active_cor_logits)
                    align_reverse_kl_loss = calculate_kl_loss(active_align_reverse_logits, active_cor_logits)
                    kl_loss = self.alpha * align_kl_loss + (1 - self.alpha) * align_reverse_kl_loss
                
            loss += self.kl_loss_weight * kl_loss
        
        return {'loss': loss, 'lm_loss': lm_loss}
        
    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.cor_bart.save_pretrained(save_path)