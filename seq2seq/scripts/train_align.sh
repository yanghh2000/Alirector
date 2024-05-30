seed=2024
src_dropout=0.2
dropout=0.1
export CUDA_VISIBLE_DEVICES=0,1

# mucgec
model_path="fnlp/bart-large-chinese"
data_path="data/MuCGEC/train_stage2_with_pred_bart.json"
    # forward alignment
accelerate launch --num_processes 2 --main_process_port 1234 \
    seq2seq/src/train_align.py \
    --model_path $model_path \
    --data_path $data_path \
    --val_set_size 1000 \
    --output_dir "seq2seq/save_ckpt/align-bart-mucgec" \
    --batch_size 512 \
    --micro_batch_size 16 \
    --max_source_length 256 \
    --max_target_length 128 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --lr_scheduler_type 'polynomial' \
    --seed $seed \
    --adam_betas '(0.9,0.999)' \
    --warmup_steps 1000 \
    --patience 5 \
    --logging_steps 100 \
    --dropout $dropout \
    --src_dropout $src_dropout

    # reverse alignment
accelerate launch --num_processes 2 --main_process_port 1234 \
    seq2seq/src/train_align.py \
    --model_path $model_path \
    --data_path $data_path \
    --val_set_size 1000 \
    --output_dir "seq2seq/save_ckpt/align-bart-mucgec-reverse" \
    --batch_size 512 \
    --micro_batch_size 16 \
    --max_source_length 256 \
    --max_target_length 128 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --lr_scheduler_type 'polynomial' \
    --seed $seed \
    --adam_betas '(0.9,0.999)' \
    --warmup_steps 1000 \
    --patience 5 \
    --logging_steps 100 \
    --dropout $dropout \
    --src_dropout $src_dropout \
    --input_reverse True \

# fcgec
model_path="seq2seq/save_ckpt/align-bart-mucgec/best-model"
data_path="data/FCGEC/FCGEC_train_with_pred_bart.json"
    # forward alignment
accelerate launch --num_processes 2 --main_process_port 1234 \
    seq2seq/src/train_align.py \
    --model_path $model_path \
    --data_path $data_path \
    --val_set_size 1000 \
    --output_dir "seq2seq/save_ckpt/align-bart-fcgec" \
    --batch_size 512 \
    --micro_batch_size 16 \
    --max_source_length 256 \
    --max_target_length 128 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --lr_scheduler_type 'polynomial' \
    --seed $seed \
    --adam_betas '(0.9,0.999)' \
    --warmup_steps 1000 \
    --patience 5 \
    --logging_steps 5 \
    --dropout $dropout \
    --src_dropout $src_dropout

    # reverse alignment
model_path="seq2seq/save_ckpt/align-bart-mucgec-reverse"
accelerate launch --num_processes 2 --main_process_port 1234 \
    seq2seq/src/train_align.py \
    --model_path $model_path \
    --data_path $data_path \
    --val_set_size 1000 \
    --output_dir "seq2seq/save_ckpt/align-bart-fcgec-reverse" \
    --batch_size 512 \
    --micro_batch_size 16 \
    --max_source_length 256 \
    --max_target_length 128 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --lr_scheduler_type 'polynomial' \
    --seed $seed \
    --adam_betas '(0.9,0.999)' \
    --warmup_steps 1000 \
    --patience 5 \
    --logging_steps 5 \
    --dropout $dropout \
    --src_dropout $src_dropout \
    --input_reverse True \