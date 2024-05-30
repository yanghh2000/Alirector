import os
from multiprocessing.pool import Pool
import time
from functools import partial
import fire
import math

def split_input(input_path, output_path, num_splits=4):
    with open(input_path, 'r', encoding='utf-8') as f:
        input_texts = f.readlines()
    
    input_paths = []
    output_paths = []
    batch_size = math.ceil(len(input_texts) / num_splits)
    for i in range(num_splits):
        _input_path = f'{input_path}.{i}.of{num_splits}'
        _output_path = f'{output_path}.{i}.of{num_splits}'
        with open(_input_path, 'w', encoding='utf-8') as o:
            o.writelines(input_texts[i*batch_size:(i+1)*batch_size])
            
        input_paths.append(_input_path)
        output_paths.append(_output_path)
    
    return input_paths, output_paths

def merge_output(output_path, num_splits=4):
    all = []
    for i in range(num_splits):
        with open(f'{output_path}.{i}.of{num_splits}', 'r', encoding='utf-8') as f:
            all.extend([line.strip() for line in f.readlines()])
            
    with open(output_path, 'w', encoding='utf-8') as o:
        o.write('\n'.join(all))

def fun(args, base_model, lora_weights, batch_size, num_beams, prompt_template, copy):
    rank, input_path, output_path = args
    cmd = f'CUDA_VISIBLE_DEVICES={rank} python evaluate_cgec.py \
        --input_path {input_path} --output_path {output_path} \
        --lora_weights {lora_weights} --base_model {base_model} --batch_size {batch_size} --num_beams {num_beams} --prompt_template {prompt_template} \
        --copy {copy}'
    print(cmd)
    os.system(cmd)

def main(
    lora_weights='models/llm/save/baichuan2-7b-base-r-8-3epoch/best-model',
    base_model='plm/Baichuan2/baichuan-inc/Baichuan2-7B-Base',
    batch_size:int = 20,
    num_beams: int = 1, 
    input_path: str = 'data/NLPCC/test/nlpcc2018_test_source.txt',
    output_path: str = 'models/llm/save/baichuan2-7b-base-r-8-3epoch/best-model/nlpcc2018_test_pred.txt',
    num_splits: int = 4,
    prompt_template: str = 'llm/templates/baichuan_prompt.json',
    copy: bool = False,
    gpu_rank_start: int = 4,
    gpu_rank_end: int = 8,
):
    input_paths, output_paths = split_input(input_path, output_path, num_splits=num_splits)
    # main
    partial_fun = partial(fun, lora_weights=lora_weights, base_model=base_model, batch_size=batch_size, num_beams=num_beams, prompt_template=prompt_template, copy=copy)
    pool = Pool()
    for rank, _input_path, _output_path in zip(range(gpu_rank_start, gpu_rank_end), input_paths, output_paths): 
        args = rank, _input_path, _output_path
        pool.apply_async(partial_fun, (args,))
    pool.close()
    pool.join()
    merge_output(output_path, num_splits=num_splits)
    
if __name__ == '__main__':
    fire.Fire(main)