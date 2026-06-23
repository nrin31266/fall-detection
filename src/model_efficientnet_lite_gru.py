import torch
import torch.nn as nn
import timm


class EfficientNetLiteGRU(nn.Module):
    """
    EfficientNet-Lite0 + GRU for video fall detection.

    Input:
        x: [B, T, C, H, W], pixel range [0, 1]

    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnet_lite0",
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # ImageNet normalization for pretrained EfficientNet.
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        """
        x: [B, T, C, H, W], pixel range [0, 1]
        """
        b, t, c, h, w = x.shape

        # [B, T, C, H, W] -> [B*T, C, H, W]
        x = x.view(b * t, c, h, w)

        # Normalize before pretrained backbone.
        x = (x - self.mean) / self.std

        # [B*T, feature_dim]
        features = self.backbone(x)

        # [B*T, feature_dim] -> [B, T, feature_dim]
        features = features.view(b, t, -1)

        # [B, T, hidden_size]
        gru_out, _ = self.gru(features)

        # Use last time step.
        last_hidden = gru_out[:, -1, :]

        logits = self.classifier(last_hidden)

        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EfficientNetLiteGRU(
        pretrained=False,
        freeze_backbone=True,
    ).to(device)

    x = torch.randn(2, 16, 3, 224, 224).to(device)

    with torch.no_grad():
        y = model(x)

    print("Device:", device)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output:", y)