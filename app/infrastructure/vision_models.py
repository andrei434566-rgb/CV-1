"""Allow-listed discriminative CNN architectures; no language/generative models."""
from __future__ import annotations

ARCHITECTURES = ('efficientnet_b0', 'convnext_tiny')
LABELS = ('sandstone', 'limestone', 'shale', 'garbage')
LABELS_RU = ('Песчаник', 'Известняк', 'Глинистая порода', 'Фон / не керн')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def create_network(architecture: str, classes: int = 4):
    from torch import nn
    from torchvision.models import efficientnet_b0, convnext_tiny
    if architecture == 'efficientnet_b0':
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    elif architecture == 'convnext_tiny':
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, classes)
    else:
        raise ValueError('Unsupported non-generative architecture: '+architecture)
    return model

def pretrained_network(architecture: str, cache):
    import torch
    from torchvision.models import efficientnet_b0, convnext_tiny, EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights
    from torch import nn
    torch.hub.set_dir(str(cache))
    if architecture == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
    elif architecture == 'convnext_tiny':
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(LABELS))
    else:
        raise ValueError(architecture)
    return model
