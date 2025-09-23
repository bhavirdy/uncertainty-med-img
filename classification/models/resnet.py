import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50(num_classes, weights=ResNet50_Weights.DEFAULT, dropout_p=0.5):
    model = resnet50(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the fully connected layer with Dropout + Linear
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Make sure fc is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
