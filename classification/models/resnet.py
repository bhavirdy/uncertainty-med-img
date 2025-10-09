import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT, dropout_p=0.5):
        super().__init__()
        self.model = resnet50(weights=weights)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the fully connected layer with Dropout + Linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

        # Make sure fc is trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
