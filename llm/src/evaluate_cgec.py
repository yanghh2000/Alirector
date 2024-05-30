import os
import sys
import gradio as gr
import torch
import transformers
import bitsandbytes as bnb
from peft import PeftModel
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from utils.callbacks import Iteratorize, Stream
from utils.prompter import CGECPrompter
import fire

from typing import Optional, List
from tqdm import tqdm

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = '',
    prompt_template: str = "llm/templates/baichuan_prompt.json",  
    rank: Optional[int] = None,
    file_index: Optional[int] = None,
    batch_size: int = 4,
    use_fast: bool = False,
    input_path: str = '',
    output_path: str = '',
    num_beams: int = 1,
    copy: bool = False,
):
    device_map = {"": 0}
    
    if file_index is None and rank is not None:
        file_index = rank 
    prompter = CGECPrompter(prompt_template)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              trust_remote_code=True,
                                              padding_side = 'left',
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
        load_in_8bit=load_8bit, 
        **model_kwargs,
    )
    model.config.use_cache = True
    print(model.__class__.__name__)
    
    
    def model_pred(model, file_index=0, lora_weight=None):
        print('lora weight: ', lora_weight)
        os.makedirs(lora_weight, exist_ok=True)
        if lora_weight and lora_weight != 'none':
            model = PeftModel.from_pretrained(
                model,
                lora_weight,
                torch_dtype=dtype,
            )

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.
            
        model.eval()
            
        print(output_path)
        cgec_predict(input_path, output_path, batch_size=batch_size)

    def batch_predict(
        texts=None,
        max_input_length=128,
        temperature=0.7,
        top_p=0.8,
        top_k=10,
        max_new_tokens=512,
        repetition_penalty=1.0,
    ):
        prompts = [prompter.generate_prompt(source=text, bos_token=tokenizer.bos_token)
                  for text in texts]
        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k:v.cuda() for k, v in inputs.items()}
        generation_config = GenerationConfig(
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
        )
        with torch.no_grad():
            generation_output = model.generate(**inputs, generation_config=generation_config)
        sequences = generation_output.sequences
        outputs = tokenizer.batch_decode(sequences.detach().cpu(), skip_special_tokens=True)
        responses = [prompter.get_response(output) for output in outputs]
        return responses

    def cgec_predict(
        input_path: str = '',
        output_path: str = '',
        batch_size=32,
    ):
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        res = []
        batch_size = batch_size // num_beams
        for idx in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[idx:idx+batch_size]
            corr_texts = batch_predict(batch_texts)
            corr_texts = [line.replace(' ', '') for line in corr_texts]
            # print(corr_texts[0])
            res.extend(corr_texts)
        
        with open(output_path, 'w', encoding='utf-8') as o:
            o.write('\n'.join(res))
    
    # main
    model_pred(model, file_index, lora_weights)

if __name__ == "__main__":
    fire.Fire(main)