import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision.models.quantization import resnet50

def setup_classifier_head(
        num_ftrs: int,
        num_immediate_features: int,
        len_class_names: int,
        dropout_rate: float) -> nn.Sequential:
    
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, num_immediate_features),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(num_immediate_features, len_class_names)
    )

def load_model(
        backbone: str,
        num_immediate_features: int, 
        num_class: int,
        dropout_rate: float) -> nn.Module:
    
    if backbone == "efficientnet_v2s":
        model_conv = efficientnet_v2_s(weights='IMAGENET1K_V1')
        for param in model_conv.parameters():
            param.requires_grad = False
        num_ftrs = model_conv.classifier[1].in_features
        model_conv.classifier = setup_classifier_head(num_ftrs, num_immediate_features, num_class, dropout_rate)
    elif backbone == "resnet50":
        model_conv = resnet50(weights='IMAGENET1K_V1', quantize=False)
        for param in model_conv.parameters():
            param.requires_grad = False
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = setup_classifier_head(num_ftrs, num_immediate_features, num_class, dropout_rate)
    else:
        raise ValueError("Invalid backbone")
    return model_conv