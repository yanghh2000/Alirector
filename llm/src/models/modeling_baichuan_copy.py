from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any, List
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from seq2seq.src.models.modeling_copy import CopyMechModule
from baichuan2.modeling_baichuan import BaichuanForCausalLM
from baichuan2.configuration_baichuan import BaichuanConfig
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

@dataclass
class CausalLMOutputWithSrcIds(CausalLMOutputWithPast):
    
    src_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    src_input_ids: Optional[Tuple[torch.LongTensor]] = None

class BaichuanForCausalLMWithCopy(BaichuanForCausalLM):
    def __init__(self, config: BaichuanConfig):
        setattr(config, 'copy', True)
        super().__init__(config)
        self.copy_module = CopyMechModule(config.hidden_size, config.vocab_size)
        
        # Initialize weights and apply final processing
        self.post_init()

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder)
        model_kwargs["src_hidden_states"] = outputs.src_hidden_states
        model_kwargs["src_input_ids"] = outputs.src_input_ids
        return model_kwargs
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "src_hidden_states": kwargs.get("src_hidden_states", None),
                "src_input_ids": kwargs.get("src_input_ids", None),
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def get_cross_attentions(self, attentions: torch.FloatTensor, labels: torch.LongTensor):
        '''
        Keep only cross-attention scores, set self-attention scores to 0.0,
        attentions: [batch_size, seq_len, seq_len]
        labels: [batch_size, seq_len]
        '''
        bsz, tgt_len, src_len = attentions.size()
        is_cross_attention = (labels[:, None, :] == -100).expand(bsz, tgt_len, src_len).to(torch.bool)
        attentions[~is_cross_attention] = 0.0
        return attentions
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        src_input_ids: Optional[torch.LongTensor] = None,
        src_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithSrcIds]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        
        # copy mechanism
        attentions = outputs.attentions[-1].mean(dim=1)
        if labels is not None:  # Training
            cross_attentions = self.get_cross_attentions(attentions, labels)
            # cross_attentions = attentions       # include targer2target attention   # TODO: exclude target2target attention
            p_gen, cp_logits = self.copy_module.forward(input_ids, cross_attentions, hidden_states, hidden_states)
            p_copy = 1 - p_gen
            logits = p_gen * lm_logits + p_copy * cp_logits
        elif src_hidden_states is None:  # Inference (Parallel Encoding)
            logits = lm_logits
            src_hidden_states = hidden_states
            src_input_ids = input_ids
        else:    # Inference (Step-by-step Decoding)
            cross_attentions = attentions[:, :, :src_hidden_states.size(1)]
            p_gen, cp_logits = self.copy_module.forward(src_input_ids, cross_attentions, src_hidden_states, hidden_states)
            p_copy = 1 - p_gen
            logits = p_gen * lm_logits + p_copy * cp_logits
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            softmax_normalizer = shift_logits.max(-1).values ** 2
            z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) + z_loss

        if not return_dict:
            output = (logits,) + outputs[1:] + (src_hidden_states, )
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithSrcIds(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            src_hidden_states=src_hidden_states,
            src_input_ids=src_input_ids
        )