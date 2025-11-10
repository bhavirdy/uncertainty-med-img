import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import Bottleneck

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

class BottleneckWithMC(nn.Module):
    def __init__(self, block, dropout_p=0.3):
        super().__init__()
        self.block = block
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(self, x):
        out = self.block(x)
        out = self.dropout(out)
        return out

class ResNet50MCDO(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT, dropout_p=0.3):
        super().__init__()
        self.model = resnet50(weights=weights)
        
        # Freeze pre-trained backbone
        for p in self.model.parameters():
            p.requires_grad = False

        # Inject Dropout into Bottleneck blocks
        self.model.layer2 = nn.Sequential(*[
            BottleneckWithMC(block, dropout_p=dropout_p)
            for block in self.model.layer2
        ])
        self.model.layer3 = nn.Sequential(*[
            BottleneckWithMC(block, dropout_p=dropout_p)
            for block in self.model.layer3
        ])
        self.model.layer4 = nn.Sequential(*[
            BottleneckWithMC(block, dropout_p=dropout_p)
            for block in self.model.layer4
        ])

        # Replace final fully connected layer with dropout + linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

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
