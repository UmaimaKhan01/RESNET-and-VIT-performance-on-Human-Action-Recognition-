
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit_b16(num_classes: int, pretrained: bool = True, dropout: float = 0.0):
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
    else:
        model = vit_b_16(weights=None)

    in_features = model.heads.head.in_features
    if dropout > 0:
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.heads.head = nn.Linear(in_features, num_classes)
    return model
