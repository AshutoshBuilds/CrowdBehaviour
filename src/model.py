import torch
import torch.nn as nn
import torchvision.models as models


class BackboneWrapper(nn.Module):
    """
    Unifies feature extraction across different torchvision backbones.
    Returns a flattened feature vector of size `out_dim`.
    """

    def __init__(self, module: nn.Module, out_dim: int):
        super().__init__()
        self.module = module
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.module(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        return feats


def _build_resnet50(pretrained: bool) -> BackboneWrapper:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = models.resnet50(weights=weights)
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules)
    return BackboneWrapper(backbone, 2048)


def _build_efficientnet_b0(pretrained: bool) -> BackboneWrapper:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    eff = models.efficientnet_b0(weights=weights)
    out_dim = eff.classifier[1].in_features
    eff.classifier = nn.Identity()
    return BackboneWrapper(eff, out_dim)


def _build_vit_b16(pretrained: bool) -> BackboneWrapper:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    vit = models.vit_b_16(weights=weights)
    out_dim = vit.heads.head.in_features
    vit.heads = nn.Identity()
    return BackboneWrapper(vit, out_dim)


def _build_densenet121(pretrained: bool) -> BackboneWrapper:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    densenet = models.densenet121(weights=weights)
    out_dim = densenet.classifier.in_features
    densenet.classifier = nn.Identity()
    return BackboneWrapper(densenet, out_dim)


def _build_mobilenet_v3_large(pretrained: bool) -> BackboneWrapper:
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    mobilenet = models.mobilenet_v3_large(weights=weights)
    # Keep everything until the last linear to output a 1280-dim vector
    truncated = list(mobilenet.classifier.children())[:-1]
    mobilenet.classifier = nn.Sequential(*truncated)
    out_dim = 1280
    return BackboneWrapper(mobilenet, out_dim)


def _build_custom_cnn(_: bool) -> BackboneWrapper:
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.proj = nn.Linear(256, 512)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.proj(x)

    model = SmallCNN()
    return BackboneWrapper(model, 512)


BACKBONE_BUILDERS = {
    "resnet50": _build_resnet50,
    "resnet": _build_resnet50,
    "efficientnet_b0": _build_efficientnet_b0,
    "efficientnet": _build_efficientnet_b0,
    "vit_b_16": _build_vit_b16,
    "vit": _build_vit_b16,
    "densenet121": _build_densenet121,
    "densenet": _build_densenet121,
    "mobilenet_v3_large": _build_mobilenet_v3_large,
    "mobilenet": _build_mobilenet_v3_large,
    "custom_cnn": _build_custom_cnn,
    "custom": _build_custom_cnn,
}


def build_backbone(name: str, pretrained: bool = True) -> BackboneWrapper:
    key = name.lower()
    if key not in BACKBONE_BUILDERS:
        available = ", ".join(sorted(BACKBONE_BUILDERS.keys()))
        raise ValueError(f"Backbone '{name}' not supported. Choose from: {available}")
    return BACKBONE_BUILDERS[key](pretrained)


class CNNLSTM(nn.Module):
    """
    Video classifier: frame backbone -> LSTM -> classifier head.
    Supports multiple CNN/ViT backbones for fair comparison.
    """

    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        backbone: str = "resnet50",
        train_backbone: bool = False,
        pretrained_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = build_backbone(backbone, pretrained=pretrained_backbone)
        self.feature_dim = self.backbone.out_dim

        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        features = self.backbone(c_in)  # (batch*seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)