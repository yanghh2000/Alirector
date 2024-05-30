seed=2024
src_dropout=0.2
dropout=0.1
export CUDA_VISIBLE_DEVICES=0,1

model_path="fnlp/bart-large-chinese"
data_path="data/MuCGEC/train_stage1.json"

accelerate launch --num_processes 2 --main_process_port 1236 \
    seq2seq/src/train.py \
    --model_path $model_path \
    --data_path $data_path \
    --val_set_size 2000 \
    --output_dir "seq2seq/save_ckpt/bart-mucgec-stage1" \
    --batch_size 1024 \
    --micro_batch_size 32 \
    --eval_batch_size 16 \
    --max_source_length 128 \
    --max_target_length 128 \
    --eval_max_source_length 256 \
    --eval_max_target_length 256 \
    --num_train_epochs 10 \
    --seed $seed \
    --lr_scheduler_type 'polynomial' \
    --learning_rate 3e-5 \
    --warmup_steps 1000 \
    --logging_steps 100 \
    --dropout $dropout \
    --src_dropout $src_dropout