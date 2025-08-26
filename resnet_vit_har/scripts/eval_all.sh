
#!/usr/bin/env bash
set -e
python src/eval.py --data_root human-action-recognition --checkpoint artifacts/resnet18_runs/*/best.pt --split test --out logs/resnet18_test_metrics.json
python src/eval.py --data_root human-action-recognition --checkpoint artifacts/vit_b16_runs/*/best.pt  --split test --out logs/vit_b16_test_metrics.json
