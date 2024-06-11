<div align="center">
<h1>Alirector: Alignment-Enhanced Chinese Grammatical Error Corrector (Findings of ACL 2024)</h1>
</div>

## Environment
To install the environment, run:

```
pip install -r requirements.txt
```

## Data

### Download

MuCGEC and NLPCC18: download links can be found in the [MuCGEC repository](https://github.com/HillZhang1999/MuCGEC).

FCGEC: [FCGEC repository](https://github.com/xlxwalex/FCGEC).

NaCGEC: [NaCGEC repository](https://github.com/masr2000/NaCGEC).

### Process

Process the data into the same format as `data/MuCGEC/train_examples.json`.

Using `data/MuCGEC/utils.py`to split the data into two parts for two-stage training.


## Download Pretrained Models
Chinese BART large: [Hugging Face Link](https://huggingface.co/fnlp/bart-large-chinese)

Baichuan2-7B-Base: [Hugging Face Link](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)


## Training

### Initial Correction Model (Stage 1 Data)

```
# bart
bash seq2seq/scripts/train_stage1.sh

# baichuan2
bash llm/scripts/train_stage1.sh
```

### Generate Prediction for Stage 2 Data

```
# bart
bash seq2seq/scripts/generate_stage2_pred.sh

# baichuan2
bash llm/scripts/generate_stage2_pred.sh
```

### Alignment Model (Stage 2 Data)

```
# bart
bash seq2seq/scripts/train_align.sh

# baichuan2
bash llm/scripts/train_align.sh
```

### Alignment Distillation (Stage 2 Data)

```
# bart
bash seq2seq/scripts/train_alignment_distill.sh

# baichuan2
bash llm/scripts/train_alignment_distall.sh
```

## Predict and Evaluate
For predicting, please use `llm/src/predict.py` or `seq2seq/src/predict.py`.

For evaluation, we adopt the [ChERRANT scorer](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers) to calculate character-level P/R/F0.5 for FCGEC and NaCGEC, and [M2Scorer](https://github.com/nusnlp/m2scorer) to calculate word-level P/R/F0.5 for NLPCC18-Test. For the usage, please refer to [this script](https://github.com/HillZhang1999/MuCGEC/blob/main/scorers/ChERRANT/evaluate.sh).
