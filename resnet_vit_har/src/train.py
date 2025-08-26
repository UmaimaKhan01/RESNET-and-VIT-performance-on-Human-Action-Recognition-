
import os, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from data import build_dataloaders
from utils import set_seed, ensure_dir, timestamp, CSVLogger, save_checkpoint, accuracy, count_parameters
from models import build_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Root with data/train and data/test")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vit_b16"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=None, help="If None, default per model")
    p.add_argument("--weight-decay", type=float, default=None, help="If None, default per model")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32","fp16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--outdir", type=str, default="artifacts/run")
    p.add_argument("--logdir", type=str, default="logs")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        img_size=args.img_size
    )

    num_classes = len(classes)
    model = build_model(args.model, num_classes=num_classes, pretrained=True, dropout=args.dropout)
    model = model.to(device)

    # Defaults per model
    if args.lr is None:
        args.lr = 1e-3 if args.model == "resnet18" else 5e-5
    if args.weight_decay is None:
        args.weight_decay = 1e-4 if args.model == "resnet18" else 5e-2

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    exp_name = f"{args.model}_{timestamp()}"
    outdir = ensure_dir(args.outdir)
    outdir = ensure_dir(os.path.join(outdir, exp_name))
    logdir = ensure_dir(os.path.join(args.logdir, exp_name))

    csv_logger = CSVLogger(os.path.join(logdir, "training_log.csv"),
                           fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc","lr","time_sec"])

    scaler = GradScaler(enabled=(args.precision=='fp16'))

    best_val = -1.0
    print(f"Model params: {count_parameters(model):,}")

    for epoch in range(1, args.epochs+1):
        # ---- Train ----
        model.train()
        t0 = time.time()
        total, correct, running_loss = 0, 0, 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(args.precision=='fp16')):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            acc1 = accuracy(logits, targets, topk=(1,))[0]
            correct += (acc1/100.0)*images.size(0)
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ---- Validate ----
        model.eval()
        val_total, val_correct, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                with autocast(enabled=(args.precision=='fp16')):
                    logits = model(images)
                    loss = criterion(logits, targets)
                val_loss_sum += loss.item() * images.size(0)
                acc1 = accuracy(logits, targets, topk=(1,))[0]
                val_correct += (acc1/100.0)*images.size(0)
                val_total += images.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        csv_logger.log({'epoch': epoch, 'train_loss': f"{train_loss:.4f}", 'train_acc': f"{train_acc:.2f}",
                        'val_loss': f"{val_loss:.4f}", 'val_acc': f"{val_acc:.2f}",
                        'lr': f"{current_lr:.6f}", 'time_sec': f"{elapsed:.2f}"})
        print(f"[{epoch:03d}/{args.epochs}] train_acc={train_acc:.2f} val_acc={val_acc:.2f} time={elapsed:.1f}s")

        is_best = val_acc > best_val
        if is_best:
            best_val = val_acc
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'classes': classes,
            'args': vars(args)
        }, is_best, outdir=outdir)

    print(f"Training done. Best val acc: {best_val:.2f}. Checkpoints in: {outdir}")

if __name__ == "__main__":
    main()
