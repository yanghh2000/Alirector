import os
import torch
from peft import PeftModel
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from utils.prompter import CGECPrompter
import fire
import json
from typing import Optional, List
from tqdm import tqdm
from llm.src.models.modeling_baichuan_copy import BaichuanForCausalLMWithCopy

def main(
    base_model: str = "",
    lora_weights: str = '',
    prompt_template: str = "llm/templates/baichuan_prompt.json", 
    batch_size: int = 4,
    input_path: str = '',
    output_path: str = '',
    num_beams: int = 2,
    load_8bit: bool = False,
    copy: bool = False,
):
    prompter = CGECPrompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              trust_remote_code=True,
                                              padding_side = 'left',
                                              use_fast=False)
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
        device_map={"": 0},
        trust_remote_code=True,
    )
    
    model_cls = BaichuanForCausalLMWithCopy if copy else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        **model_kwargs,
    )
    model.config.use_cache = True
    
    def model_pred(model, lora_weight=None):
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
        generation_config=None,
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
        max_input_length=128,
        temperature=0.7,
        top_p=0.8,
        top_k=10,
        max_new_tokens=512,
        repetition_penalty=1.0,
    ):
        generation_config = GenerationConfig(
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
        )
        
        if tokenizer.__class__.__name__ == 'Qwen2Tokenizer':
            generation_config.pad_token_id = generation_config.eos_token_id = tokenizer.eos_token_id
        
        # load data
        if input_path.endswith('.json'):
            data = json.load(open(input_path))
            texts = [line['source'] for line in data]
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                texts = [line.strip().split('\t')[-1] for line in f.readlines()]
        
        # predict
        res = []
        batch_size = batch_size // num_beams
        for idx in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[idx:idx+batch_size]
            corr_texts = batch_predict(batch_texts, generation_config=generation_config)
            corr_texts = [line.replace(' ', '') for line in corr_texts]
            res.extend(corr_texts)
        
        # save results
        if output_path.endswith('.json'):
            for line, pred in zip(data, res):
                line['pred'] = pred
            
            with open(output_path, 'w', encoding='utf-8') as o:
                json.dump(data, o, ensure_ascii=False, indent=1)
        else:
            with open(output_path, 'w', encoding='utf-8') as o:
                o.write('\n'.join(res))
    
    # main
    model_pred(model, lora_weights)

if __name__ == "__main__":
    fire.Fire(main)