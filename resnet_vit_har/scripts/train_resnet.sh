
#!/usr/bin/env bash
set -e
python src/train.py --data_root human-action-recognition --model resnet18 --epochs 15 --batch-size 64 --precision fp16 --logdir logs --outdir artifacts/resnet18_runs
