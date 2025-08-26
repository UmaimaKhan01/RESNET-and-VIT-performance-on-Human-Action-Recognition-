
import os, argparse, json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from models import build_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--split', type=str, default='test', choices=['test','val','train'])
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--out', type=str, default='logs/eval_metrics.json')
    return p.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get('classes', None)
    if classes is None:
        raise RuntimeError('Classes not found in checkpoint. Re-train with provided scripts.')

    model_args = ckpt.get('args', {})
    model_name = model_args.get('model', 'resnet18')
    model = build_model(model_name, num_classes=len(classes), pretrained=False, dropout=model_args.get('dropout', 0.0))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    split_dir = os.path.join(args.data_root, 'data', args.split)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(split_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    all_preds, all_targs = [], []
    with torch.no_grad():
        for images, targets in dl:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targs.extend(targets.tolist())

    report = classification_report(all_targs, all_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targs, all_preds)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, 'w') as f:
        json.dump({
            'top1_acc': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'per_class': {c: report[c] for c in classes},
        }, f, indent=2)

    df = pd.DataFrame(report).transpose()
    df.to_csv(args.out.replace('.json', '_classification_report.csv'), index=True)

    import numpy as np
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(args.out.replace('.json', '_confusion_matrix.csv'))

    print(f'Saved metrics to {args.out}')

if __name__ == '__main__':
    main()
