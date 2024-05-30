export CUDA_VISIBLE_DEVICES=0

# mucgec
model_path="seq2seq/save_ckpt/bart-mucgec-stage1/best-model"
input_path="data/MuCGEC/train_stage2.json"
output_path="data/MuCGEC/train_stage2_with_pred_bart.json"

python seq2seq/src/predict.py --model_path $model_path --input_path $input_path --output_path $output_path

# fcgec
model_path="seq2seq/save_ckpt/bart-mucgec/best-model"
input_path="data/MuCGEC/FCGEC_train.json"
output_path="data/FCGEC/FCGEC_train_with_pred_bart.json"

python seq2seq/src/predict.py --model_path $model_path --input_path $input_path --output_path $output_path