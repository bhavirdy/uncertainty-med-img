import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50(num_classes, weights=ResNet50_Weights.DEFAULT, finetune_layers=("layer3", "layer4", "fc"), dropout_p=0.5):
    model = resnet50(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze selected layers
    for layer_name in finetune_layers:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True

    # Replace the fully connected layer with Dropout + Linear
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model
