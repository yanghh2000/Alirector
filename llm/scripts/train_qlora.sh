lora_r=8
lora_alpha=16
lora_dropout=0.05
base_model="baichuan-inc/Baichuan2-7B-Base"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# mucgec
data_path="data/MuCGEC/train.json"

accelerate launch --num_processes 8 --main_process_port 1235 \
    llm/src/train.py \
    --base_model $base_model \
    --data_path $data_path \
    --val_set_size 2000 \
    --output_dir "llm/save_ckpt/baichuan2-mucgec" \
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

# fcgec
resume_from_checkpoint="llm/save_ckpt/baichuan2-mucgec/best-model"
data_path="data/FCGEC/FCGEC_train.json"

accelerate launch --num_processes 8 --main_process_port 1235 \
    llm/src/train.py \
    --base_model $base_model \
    --data_path $data_path \
    --val_set_size 1000 \
    --output_dir "llm/save_ckpt/baichuan2-fcgec" \
    --batch_size 256 \
    --micro_batch_size 16 \
    --num_epochs 10 \
    --cutoff_len 150 \
    --lora_target_modules "[W_pack, o_proj, gate_proj, down_proj, up_proj]" \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --learning_rate 3e-5 \
    --prompt_template_name "llm/templates/baichuan_prompt.json" \
    --logging_steps 5 \
    --optim "paged_adamw_8bit" \
    --load_in_4bit True \
    --resume_from_checkpoint $resume_from_checkpoint \