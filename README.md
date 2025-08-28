# Assignment 1: ResNet vs Vision Transformer for Human Action Recognition

## Overview
This project compares **ResNet-18** and **ViT-B/16** on a Human Action Recognition dataset (~12.6k images, 15 classes). The focus is on both performance and computational trade-offs between CNNs and Transformers.

- **ResNet-18** → lightweight, fast, efficient; excels in motion-heavy actions.  
- **ViT-B/16** → heavy, slower, more accurate; stronger on subtle or static actions.

---

## Environment Setup
Experiments were run in a conda environment `gpu_env`:

```bash
conda create -n gpu_env python=3.10 -y
conda activate gpu_env
pip install -r requirements.txt
```

**Key packages**: `torch`, `torchvision`, `timm`, `albumentations`, `scikit-learn`, `matplotlib`, `opencv-python`.

---

## Dataset

* 15 action classes (cycling, texting, running, etc.).
* ~10,710 train images, ~1,890 test images.
* Images resized to 224×224.
* Augmentations: `RandomResizedCrop`, `RandomHorizontalFlip`, `AutoAugment`.

---

## Training Details

* **Optimizer**: AdamW with cosine decay  
* **Loss**: Cross-entropy  
* **Precision**: Mixed FP16  
* **Epochs**: 15  
* **Batch sizes**: 64 (ResNet), 32 (ViT)  
* **Learning rates**: 1e-3 (ResNet), 5e-5 (ViT)  
* **Checkpointing**: Best model saved by validation accuracy  

---

## CLI Commands

### 1. Activate Environment
```powershell
C:/Users/umaim/anaconda3/Scripts/activate
conda activate gpu_env
```

### 2. Navigate to Project
```powershell
cd .\Structured\
cd .\resnet_vit_har\
```

### 3. ResNet-18 Evaluation
```powershell
$ckpt_res = (Get-ChildItem artifacts\resnet18 -Recurse -Filter best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python src/eval.py --data_root "C:\Users\umaim\Downloads\archive\Structured" --checkpoint "$ckpt_res" --split test --out logs\resnet18_test_metrics.json
```

Inspect:
```powershell
Get-Content .\logs\resnet18_test_metrics.json
```

### 4. ViT-B/16 Evaluation
```powershell
$ckpt_vit = (Get-ChildItem artifacts\vit_b16 -Recurse -Filter best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python src/eval.py --data_root "C:\Users\umaim\Downloads\archive\Structured" --checkpoint "$ckpt_vit" --split test --out logs\vit_b16_test_metrics.json
```

Inspect:
```powershell
Get-Content .\logs\vit_b16_test_metrics.json
```

---

## Results

### Accuracy

| Model     | Top-1 Acc | Macro F1 | Weighted F1 |
| --------- | --------- | -------- | ----------- |
| ResNet-18 | 78.0%     | 78.1%    | 78.1%       |
| ViT-B/16  | 84.9%     | 84.8%    | 84.8%       |

### Computational Costs

| Model     | Params | FLOPs | Latency | Memory |
| --------- | ------ | ----- | ------- | ------ |
| ResNet-18 | 11.2M  | 1.8G  | 47 ms   | 480 MB |
| ViT-B/16  | 85.8M  | 17.6G | 236 ms  | 594 MB |

### Per-Class Performance
- **ResNet-18**: excels on motion-heavy actions (cycling F1=0.97, running F1=0.86), struggles with static ones (texting F1=0.65, sitting F1=0.67).  
- **ViT-B/16**: stronger on static actions (texting F1=0.78, sitting F1=0.72) due to attention-based context modeling.

---

## Failure Analysis

* Misclassified images saved in `failure_cases_resnet/` and `failure_cases_vit/`.  
* ResNet confuses **texting → calling**, ViT fixes this.  
* ViT sometimes mislabels **hugging → laughing**.  

---

## Demo Outputs

* Confusion matrices: `resnet18_cm.png`, `vit_b16_cm.png`  
* Failure grids: `resnet_failure.png`, `vit_failure.png`  
* Misclassification deltas: `delta_bar.png`  
* Demo frames with labels: `demo_frames/`  

---

## Insights

* **ResNet-18**  
  - Small (11M params), efficient  
  - Good for edge devices  
  - Strong on high-motion actions  

* **ViT-B/16**  
  - Large (86M params), slower  
  - Best on subtle/static actions  
  - Ideal for accuracy-critical server tasks  

**Conclusion**: ResNet is efficiency-first; ViT is accuracy-first.  

---

## Reproduction Steps

1. Install dependencies via `requirements.txt`.  
2. Run CLI commands for both models.  
3. Check logs in `logs/`.  
4. Inspect confusion matrices and failure cases.  

---
