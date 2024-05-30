seed=2024
src_dropout=0.2
dropout=0.1
export CUDA_VISIBLE_DEVICES=0,1

# mucgec
model_path="fnlp/bart-large-chinese"
data_path="data/MuCGEC/train.json"

accelerate launch --num_processes 2 --main_process_port 1236 \
    seq2seq/src/train.py \
    --model_path $model_path \
    --data_path $data_path \
    --val_set_size 2000 \
    --output_dir "seq2seq/save_ckpt/bart-mucgec" \
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

# fcgec
model_path="seq2seq/save_ckpt/bart-mucgec/best-model"
data_path="data/FCGEC/FCGEC_train.json"

accelerate launch --num_processes 2 --main_process_port 1236 \
    models/seq2seq-based-CGEC/train.py \
    --model_path $model_path \
    --data_path $data_path \
    --val_set_size 1000 \
    --output_dir "seq2seq/save_ckpt/bart-fcgec" \
    --batch_size 256 \
    --micro_batch_size 16 \
    --max_source_length 128 \
    --max_target_length 128 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --lr_scheduler_type 'polynomial' \
    --warmup_steps 100 \
    --logging_steps 5 \
    --dropout $dropout \
    --src_dropout $src_dropout