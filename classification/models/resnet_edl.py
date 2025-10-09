import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50EDL(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        self.model = resnet50(weights=weights)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final FC layer
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.Softplus()  # ensures non-negative evidence outputs
        )

        # Train FC layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        evidence = self.model(x)
        alpha = evidence + 1  # Dirichlet parameters (alpha >= 1)
        return alpha