import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50(num_classes, weights=None):
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet50_finetune(num_classes, weights=ResNet50_Weights.DEFAULT, finetune_layers=("layer3", "layer4", "fc")):
    model = resnet50(weights=weights)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specified layers
    for layer_name in finetune_layers:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True

    # Replace final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if "fc" not in finetune_layers:
        for param in model.fc.parameters():
            param.requires_grad = True  # fc should always be trainable

    return model
