import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50EDL(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT, dropout_p=0.5):
        super().__init__()
        self.model = resnet50(weights=weights)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.Softplus() # ensures non-negative evidence outputs
        )
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        evidence = self.model(x)
        alpha = evidence + 1  # Dirichlet parameters (alpha >= 1)
        return alpha