
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18(num_classes: int, pretrained: bool = True, dropout: float = 0.0):
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model
