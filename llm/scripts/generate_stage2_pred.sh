export CUDA_VISIBLE_DEVICES=0
base_model="baichuan-inc/Baichuan2-7B-Base"

# mucgec
lora_weights="llm/save_ckpt/baichuan2-mucgec-stage1/best-model"
input_path="data/MuCGEC/train_stage2.json"
output_path="data/MuCGEC/train_stage2_with_pred_baichuan2.json"

python llm/src/predict.py \
    --base_model $base_model \
    --lora_weights $lora_weights \
    --input_path $input_path \
    --output_path $output_path

# fcgec
lora_weights="llm/save_ckpt/baichuan2-mucgec/best-model"
input_path="data/MuCGEC/FCGEC_train.json"
output_path="data/FCGEC/FCGEC_train_with_pred_baichuan2.json"

python llm/src/predict.py \
    --base_model $base_model \
    --lora_weights $lora_weights \
    --input_path $input_path \
    --output_path $output_path