
import os, glob, random, argparse, cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models import build_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained checkpoint (best.pt)')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Path to test images, e.g. C:\\Users\\umaim\\Downloads\\archive\\Structured\\test')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--num', type=int, default=60,
                   help='How many frames to generate')
    p.add_argument('--out_dir', type=str, default='demo_frames',
                   help='Folder to save overlayed frames')
    p.add_argument('--make_video', type=str, default=None,
                   help='Optional mp4 output, e.g. demo_out.mp4')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt['classes']
    model_name = ckpt['args']['model']
    dropout = ckpt.get("args", {}).get("dropout", 0.0)
    model = build_model(model_name, num_classes=len(classes), pretrained=False, dropout=dropout).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Collect images
    paths = []
    for cls in os.listdir(args.data_dir):
        p = os.path.join(args.data_dir, cls)
        if os.path.isdir(p):
            paths.extend(glob.glob(os.path.join(p, '*.*')))
    random.shuffle(paths)
    paths = paths[:args.num]

    os.makedirs(args.out_dir, exist_ok=True)

    for i, pth in enumerate(paths):
        img = Image.open(pth).convert('RGB')
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()
        label = classes[pred]

        frame = cv2.cvtColor(np.array(img.resize((args.img_size, args.img_size))), cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (0,0), (args.img_size, 30), (0,0,0), -1)
        cv2.putText(frame, f'Pred: {label}', (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        out_path = os.path.join(args.out_dir, f'frame_{i:04d}.jpg')
        cv2.imwrite(out_path, frame)

    if args.make_video:
        fps = 8
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.make_video, fourcc, fps, (args.img_size, args.img_size))
        frames = sorted(glob.glob(os.path.join(args.out_dir, 'frame_*.jpg')))
        for fp in frames:
            frame = cv2.imread(fp)
            writer.write(frame)
        writer.release()
        print(f'Wrote video to {args.make_video}')

    print(f'Saved {len(paths)} frames with overlays into {args.out_dir}')

if __name__ == '__main__':
    main()
