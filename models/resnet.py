import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50(num_classes, weights=ResNet50_Weights.DEFAULT, finetune=True, lr=1e-4, layer4_lr_factor=0.1):
    model = resnet50(weights=weights)

    if finetune:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Create parameter groups for optimizer
    param_groups = [
        {'params': model.layer4.parameters(), 'lr': lr * layer4_lr_factor},
        {'params': model.fc.parameters(), 'lr': lr}
    ]

    return model, param_groups
