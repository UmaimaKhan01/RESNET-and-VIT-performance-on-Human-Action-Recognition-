
import os, math
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def build_transforms(size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def build_dataloaders(root: str,
                      batch_size: int = 64,
                      num_workers: int = 2,
                      val_split: float = 0.1,
                      img_size: int = 224):
    '''
    Expected structure under root:
      root/
        data/
          train/<class>/*
          test/<class>/*
    '''
    train_dir = os.path.join(root, "data", "train")
    test_dir  = os.path.join(root, "data", "test")

    train_tf = build_transforms(img_size, is_train=True)
    test_tf  = build_transforms(img_size, is_train=False)

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds    = datasets.ImageFolder(test_dir,  transform=test_tf)

    n_total = len(full_train)
    n_val = int(math.floor(val_split * n_total))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_train.classes
