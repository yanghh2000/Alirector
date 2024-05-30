lora_r=8
lora_alpha=16
lora_dropout=0.05
base_model="baichuan-inc/Baichuan2-7B-Base"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_path="data/MuCGEC/train_stage1.json"

accelerate launch --num_processes 8 --main_process_port 1235 \
    llm/src/train.py \
    --base_model $base_model \
    --data_path $data_path \
    --val_set_size 2000 \
    --output_dir "llm/save_ckpt/baichuan2-mucgec-stage1" \
    --batch_size 1024 \
    --micro_batch_size 16 \
    --num_epochs 3 \
    --cutoff_len 150 \
    --lora_target_modules "[W_pack, o_proj, gate_proj, down_proj, up_proj]" \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --learning_rate 3e-5 \
    --prompt_template_name "llm/templates/baichuan_prompt.json" \
    --logging_steps 20 \
    --optim "paged_adamw_8bit" \
    --load_in_4bit True