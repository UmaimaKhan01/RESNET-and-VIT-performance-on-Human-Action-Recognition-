
from .resnet18 import build_resnet18
from .vit import build_vit_b16

def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0):
    name = name.lower()
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif name in ("vit_b16", "vit-b16", "vit"):
        return build_vit_b16(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown model name: {name}")
