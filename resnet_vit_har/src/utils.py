
import os, json, csv, random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_checkpoint(state, is_best, outdir, best_name="best.pt", last_name="last.pt"):
    os.makedirs(outdir, exist_ok=True)
    torch.save(state, os.path.join(outdir, last_name))
    if is_best:
        torch.save(state, os.path.join(outdir, best_name))

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / batch_size)).item())
        return res
