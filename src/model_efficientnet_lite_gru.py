import torch
import torch.nn as nn
import timm


class EfficientNetLiteGRU(nn.Module):
    """
    EfficientNet-Lite0 + GRU for video fall detection.

    Input:
        x: [B, T, C, H, W]

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

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """
        b, t, c, h, w = x.shape

        # [B, T, C, H, W] -> [B*T, C, H, W]
        x = x.view(b * t, c, h, w)

        # [B*T, feature_dim]
        features = self.backbone(x)

        # [B*T, feature_dim] -> [B, T, feature_dim]
        features = features.view(b, t, -1)

        # GRU output: [B, T, hidden_size]
        gru_out, _ = self.gru(features)

        # lấy hidden state ở frame cuối
        last_hidden = gru_out[:, -1, :]

        logits = self.classifier(last_hidden)

        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pretrained=False để test nhanh, không cần tải weight
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