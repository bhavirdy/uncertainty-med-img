import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Deterministic(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        
        # Load pretrained ResNet50
        self.model = resnet50(weights=weights)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace FC with trainable classifier
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Make FC trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class ResNet50MCDO(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT, dropout_p=0.3):
        super().__init__()

        # Load pretrained ResNet50
        self.model = resnet50(weights=weights)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Wrap each ResNet stage with Dropout
        # self.model.layer1 = nn.Sequential(self.model.layer1, nn.Dropout2d(p=dropout_p))
        self.model.layer2 = nn.Sequential(self.model.layer2, nn.Dropout2d(p=dropout_p))
        self.model.layer3 = nn.Sequential(self.model.layer3, nn.Dropout2d(p=dropout_p))
        self.model.layer4 = nn.Sequential(self.model.layer4, nn.Dropout2d(p=dropout_p))

        # Replace FC with Dropout + Linear
        self.model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(self.model.fc.in_features, num_classes))

        # Make FC trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class ResNet50EDL(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        
        # Load pretrained ResNet50
        self.model = resnet50(weights=weights)
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.Softplus() # ensures non-negative evidence outputs
        )
        
        # Make FC trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        evidence = self.model(x)
        alpha = evidence + 1  # Dirichlet parameters (alpha >= 1)
        return alpha
