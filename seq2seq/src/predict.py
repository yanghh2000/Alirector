import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BertTokenizer, GenerationConfig, BartConfig, PreTrainedModel
from opencc import OpenCC
from typing import *
import re
import fire
from models.modeling_copy import BartForConditionalGenerationWithCopyMech
from models.modeling_bart_dropsrc import BartForConditionalGenerationwithDropoutSrc
import math

model_cls_dict = {
    'BartForConditionalGenerationWithCopyMech': BartForConditionalGenerationWithCopyMech,
    'BartForConditionalGeneration': BartForConditionalGeneration,
    'BartForConditionalGenerationwithDropoutSrc': BartForConditionalGenerationwithDropoutSrc,
}

def split_sentence(text, max_length=64):
    split_texts = []
    for idx in range(0, len(text), max_length):
        split_texts.append(text[idx:idx+max_length])
    return split_texts

def batch_split_sentence(texts, max_length=64):
    split_texts = []
    ids = []
    i = 0
    for text in texts:
        res = split_sentence(text, max_length)
        split_texts.extend(res)
        ids.append((i, i+len(res)-1))
        i += len(res)
        
    return split_texts, ids
        
def main(
    model_path: str = "",
    input_path: str = "",
    batch_size: int = 400,
    output_path: str = "",
    split_length=100,
    if_split: bool = False,     # whether or not to split long sentence into short ones before inference
    max_target_length: int = 128,
    temperature: float = 1,
    num_beams: int = 10,
    src_dropout=0.2,
):     
    cc = OpenCC("t2s")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    config = BartConfig.from_pretrained(model_path)
    model_cls:PreTrainedModel = model_cls_dict[config.architectures[0]]
    if model_cls == BartForConditionalGenerationwithDropoutSrc:
        model = BartForConditionalGenerationwithDropoutSrc.from_pretrained(model_path, src_dropout=src_dropout)
    else:
        model = model_cls.from_pretrained(model_path)
    print(model.__class__.__name__)
    
    model.half()
    model.cuda()
    model.eval()
    model.config.use_cache = True
    
    if input_path.endswith('.json'):
        data = json.load(open(input_path))
        texts = [line['source'] for line in data]
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip().split('\t')[-1] for line in f.readlines()]
    
    batch_size = batch_size // num_beams
    pred_texts = []
    for idx in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[idx:idx+batch_size]
        if if_split:
            split_texts, ids = batch_split_sentence(batch_texts, split_length)
        else:
            split_texts = batch_texts
        inputs = tokenizer(
            split_texts,
            padding=True,
            return_tensors='pt',
            return_token_type_ids=False,
        )
        inputs = {k:v.cuda() for k, v in inputs.items()}
        generation_config = GenerationConfig(
            num_beams=num_beams,  
            temperature=temperature,
            max_new_tokens=max_target_length,
        )
        with torch.no_grad():
            pred_ids = model.generate(
                **inputs,
                generation_config=generation_config,
            )
        preds = tokenizer.batch_decode(pred_ids.detach().cpu(), skip_special_tokens=True)
        preds = [pred.replace(' ', '') for pred in preds]
        
        if if_split:
            for start, end in ids:
                pred_text = ''.join(preds[start:end+1])
                pred_text = cc.convert(pred_text)
                pred_texts.append(pred_text)
        else:
            pred_texts.extend(preds)
    

    if output_path.endswith('.json'):
        for line, pred in zip(data, pred_texts):
            line['pred'] = pred
        
        with open(output_path, 'w', encoding='utf-8') as o:
            json.dump(data, o, ensure_ascii=False, indent=1)
    else:
        with open(output_path, 'w', encoding='utf-8') as o:
            o.write('\n'.join(pred_texts))

if __name__ == "__main__":
    fire.Fire(main)
