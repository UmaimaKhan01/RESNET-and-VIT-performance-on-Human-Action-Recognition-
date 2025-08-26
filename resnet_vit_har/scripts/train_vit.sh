
#!/usr/bin/env bash
set -e
python src/train.py --data_root human-action-recognition --model vit_b16 --epochs 15 --batch-size 32 --precision fp16 --lr 5e-5 --weight-decay 0.05 --logdir logs --outdir artifacts/vit_b16_runs
