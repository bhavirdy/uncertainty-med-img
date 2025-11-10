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

class DecoderBlockWithDropout(nn.Module):
    def __init__(self, decoder_block, dropout_p=0.3):
        super().__init__()
        self.decoder_block = decoder_block
        self.dropout = nn.Dropout2d(p=dropout_p)
    
    def forward(self, feature_map, target_height, target_width, skip_connection=None):
        x = self.decoder_block(feature_map, target_height, target_width, skip_connection)
        x = self.dropout(x)
        return x

class UNetMCDO(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, encoder_name='resnet34', encoder_weights='imagenet', dropout_p=0.5):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes
        )
        
        # Add dropout to all decoder blocks
        self._add_dropout_to_decoder(dropout_p)
        
        # Add dropout before final segmentation head
        self.final_dropout = nn.Dropout2d(p=dropout_p)
    
    def _add_dropout_to_decoder(self, dropout_p):
        decoder = self.unet.decoder
        
        # Wrap each decoder block with dropout
        for i in range(len(decoder.blocks)):
            original_block = decoder.blocks[i]
            decoder.blocks[i] = DecoderBlockWithDropout(original_block, dropout_p)
    
    def forward(self, x):
        # Standard forward pass through the entire UNet
        # encoder -> decoder with dropout -> segmentation head
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(features)
        decoder_output = self.final_dropout(decoder_output)
        masks = self.unet.segmentation_head(decoder_output)
        return masks

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
