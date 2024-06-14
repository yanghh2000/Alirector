import json

def split(path, stage1_ratio, stage1_path, stage2_path):
    with open(path, 'r') as f:
        data = json.load(f)
    n = len(data)
    n1 = int(n * stage1_ratio)
    data1 = data[:n1]
    data2 = data[n1:]
    with open(stage1_path, 'w') as f:
        json.dump(data1, f)
    with open(stage2_path, 'w') as f:
        json.dump(data2, f)

if __name__ == '__main__':
    split('data/MuCGEC/train.json', 0.7, 'data/MuCGEC/train_stage1.json', 'data/MuCGEC/train_stage2.json')
