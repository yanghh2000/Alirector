kl_loss_weight=0.5
src_dropout=0.2
dropout=0.1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# mucgec
cor_bart_path="seq2seq/save_ckpt/bart-mucgec-stage1/best-model"
align_bart_path="seq2seq/save_ckpt/align-bart-mucgec/best-model"
reverse_align_bart_path="seq2seq/save_ckpt/align-bart-mucgec-reverse/best-model"
data_path="data/MuCGEC/train_stage2_with_pred_bart.json"

accelerate launch --num_processes 8 --main_process_port 1233 \
    seq2seq/src/train_alignment_distill.py \
    --cor_bart_path $cor_bart_path \
    --align_bart_path $align_bart_path \
    --reverse_align_bart_path $reverse_align_bart_path \
    --data_path $data_path \
    --val_set_size 2000 \
    --output_dir "seq2seq/save_ckpt/alignment-distill-bart-mucgec" \
    --batch_size 1024 \
    --micro_batch_size 32 \
    --max_cor_length 128 \
    --max_align_length 256 \
    --max_target_length 128 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --kl_loss_weight $weight \
    --logging_steps 100 \
    --kl_loss_type "forward-kl" \
    --dropout $dropout \
    --src_dropout $src_dropout

# fcgec
cor_bart_path="seq2seq/save_ckpt/alignment-distill-bart-mucgec/best-model"
align_bart_path="seq2seq/save_ckpt/align-bart-fcgec/best-model"
reverse_align_bart_path="seq2seq/save_ckpt/align-bart-fcgec-reverse/best-model"
data_path="data/FCGEC/FCGEC_train_with_pred_bart.json"

accelerate launch --num_processes 8 --main_process_port 1233 \
    seq2seq/src/train_alignment_distill.py \
    --cor_bart_path $cor_bart_path \
    --align_bart_path $align_bart_path \
    --reverse_align_bart_path \
    --data_path $data_path \
    --val_set_size 1000 \
    --output_dir "seq2seq/save_ckpt/alignment-distill-bart-fcgec" \
    --batch_size 960 \
    --micro_batch_size 24 \
    --max_cor_length 128 \
    --max_align_length 192 \
    --max_target_length 128 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --kl_loss_weight $weight \
    --logging_steps 5 \
    --kl_loss_type "forward-kl" \
    --dropout $dropout \
    --src_dropout $src_dropout