
import argparse, time, json
import torch
from models import build_model
from utils import count_parameters

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='resnet18', choices=['resnet18','vit_b16'])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--out', type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.model, num_classes=15, pretrained=False).to(device).eval()

    params = count_parameters(model)

    flops = None
    try:
        from ptflops import get_model_complexity_info
        with torch.no_grad():
            flops, _ = get_model_complexity_info(model, (3, args.img_size, args.img_size), as_strings=False, print_per_layer_stat=False)
    except Exception:
        pass

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()

    avg_latency_ms = 1000.0 * (t1 - t0) / args.iters
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == 'cuda' else None

    out = {
        'model': args.model,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'params': int(params),
        'flops': int(flops) if flops is not None else None,
        'avg_latency_ms': round(avg_latency_ms, 3),
        'peak_mem_mb': round(peak_mem_mb, 2) if peak_mem_mb is not None else None
    }

    print(json.dumps(out, indent=2))
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()
