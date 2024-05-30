kl_loss_weight=0.5
lora_r=8
lora_alpha=16
lora_dropout=0.05
base_model="baichuan-inc/Baichuan2-7B-Base"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# mucgec
cor_lora_path="llm/save_ckpt/baichuan2-mucgec-stage1/best-model"
align_lora_path="llm/save_ckpt/align-baichuan2-mucgec/best-model"
reverse_align_lora_path="llm/save_ckpt/align-baichuan2-mucgec-reverse/best-model"
data_path="data/MuCGEC/train_stage2_with_pred_baichuan2.json"

accelerate launch --num_processes 8 --main_process_port 1234 \
        llm/src/train_llm_alignment_distill.py \
        --base_model $base_model \
        --cor_lora_path $cor_lora_path \
        --align_lora_path $align_lora_path \
        --reverse_align_lora_path $reverse_align_lora_path \
        --data_path $data_path \
        --val_set_size 2000 \
        --output_dir "llm/save_ckpt/alignment-distill-baichuan2-mucgec" \
        --batch_size 1024 \
        --micro_batch_size 32 \
        --num_epochs 3 \
        --cutoff_len 192 \
        --lora_target_modules "[W_pack, o_proj, gate_proj, down_proj, up_proj, copy_head]" \
        --lora_r $lora_r \
        --lora_alpha $lora_alpha \
        --lora_dropout $lora_dropout \
        --learning_rate 3e-5 \
        --prompt_template_name "llm/templates/baichuan_prompt.json" \
        --logging_steps 20 \
        --kl_loss_weight $kl_loss_weight \
        --kl_loss_type "forward-kl" \
        --eval_steps 0.1 \
        --early_stopping_patience 5 \
        --optim "paged_adamw_8bit"

# fcgec
cor_lora_path="llm/save_ckpt/alignment-distill-baichuan2-mucgec/best-model"
align_lora_path="llm/save_ckpt/align-baichuan2-fcgec/best-model"
reverse_align_lora_path="llm/save_ckpt/align-baichuan2-fcgec-reverse/best-model"
data_path="data/FCGEC/FCGEC_train_with_pred_baichuan2.json"

accelerate launch --num_processes 8 --main_process_port 1234 \
        llm/src/train_llm_alignment_distill.py \
        --base_model $base_model \
        --cor_lora_path $cor_lora_path \
        --align_lora_path $align_lora_path \
        --reverse_align_lora_path $reverse_align_lora_path \
        --data_path $data_path \
        --val_set_size 1000 \
        --output_dir "llm/save_ckpt/alignment-distill-baichuan2-fcgec" \
        --batch_size 256 \
        --micro_batch_size 16 \
        --num_epochs 10 \
        --cutoff_len 256 \
        --lora_target_modules "['W_pack', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']" \
        --lora_r $lora_r \
        --lora_alpha $lora_alpha \
        --lora_dropout $lora_dropout \
        --learning_rate 3e-5 \
        --prompt_template_name "llm/templates/baichuan_prompt.json" \
        --logging_steps 10 \
        --kl_loss_weight $kl_loss_weight \
        --kl_loss_type "forward-kl" \
        --eval_steps 0.1 \
        --early_stopping_patience 5 \
        --optim "paged_adamw_8bit"