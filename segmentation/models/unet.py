import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetDeterministic(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, encoder_name='resnet34', encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes
        )

    def forward(self, x):
        return self.model(x)

class UNetMCDO(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, encoder_name='resnet34', encoder_weights='imagenet', dropout_p=0.3):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes
        )

        # Add dropout after each decoder block
        for i, block in enumerate(self.model.decoder.blocks):
            block.add_module("mc_dropout", nn.Dropout2d(p=dropout_p))

    def forward(self, x):
        return self.model(x)

class UNetEDL(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, encoder_name='resnet34', encoder_weights='imagenet'):
        super().__init__()
        # Base UNet backbone
        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes
        )
        # Evidence transformation: ensures non-negative evidence
        self.evidence_head = nn.Softplus()

    def forward(self, x):
        outputs = self.backbone(x)
        evidence = self.evidence_head(outputs)
        alpha = evidence + 1  # Dirichlet concentration parameters
        return alpha
