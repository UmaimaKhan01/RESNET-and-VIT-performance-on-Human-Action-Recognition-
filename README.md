
# ResNet18 vs ViT on Human Action Recognition

Train and evaluate **ResNet18** and **ViT-B/16** on the Kaggle Human Action Recognition dataset (15 classes, ~12.6k images).
Includes CLI scripts, logs, eval outputs, and demo/video utilities.

## 1) Setup

### Conda
```bash
conda env create -f environment.yml
conda activate resnet-vit-har
```

### venv + pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Dataset

Expected layout:
```
human-action-recognition/
└── data/
    ├── train/
    │   ├── class_a/ ...
    │   └── ...
    └── test/
        ├── class_a/ ...
        └── ...
```

In Colab (if `archive.zip` is uploaded):
```python
import zipfile, os
z = zipfile.ZipFile("archive.zip")
z.extractall("human-action-recognition")
print(os.listdir("human-action-recognition"))
```

## 3) Train
```bash
python src/train.py --data_root human-action-recognition --model resnet18 --epochs 15 --batch-size 64 --precision fp16 --logdir logs --outdir artifacts/resnet18
python src/train.py --data_root human-action-recognition --model vit_b16  --epochs 15 --batch-size 32 --precision fp16 --lr 5e-5 --weight-decay 0.05 --logdir logs --outdir artifacts/vit_b16
```

## 4) Evaluate (test split)
```bash
python src/eval.py --data_root human-action-recognition --checkpoint artifacts/resnet18/best.pt --split test --out logs/resnet18_test_metrics.json
python src/eval.py --data_root human-action-recognition --checkpoint artifacts/vit_b16/best.pt  --split test --out logs/vit_b16_test_metrics.json
```

## 5) Computational metrics
```bash
python src/measure_model.py --model resnet18 --batch-size 64
python src/measure_model.py --model vit_b16  --batch-size 32
```

## 6) Logs & Outputs
- Training CSV: `logs/<exp>/training_log.csv`
- Best checkpoint: `artifacts/<exp>/best.pt`
- Eval JSON/CSV: in `logs/`
- TensorBoard: `tensorboard --logdir logs`

## 7) Shell shortcuts
```bash
bash scripts/train_resnet.sh
bash scripts/train_vit.sh
bash scripts/eval_all.sh
```
