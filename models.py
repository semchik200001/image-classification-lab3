"""Архитектуры сетей.

  cnn              — своя небольшая свёрточная сеть, обучается с нуля
  resnet18         — ResNet18, предобученная на ImageNet, дообучается целиком
  resnet18_frozen  — ResNet18, обучается только последний слой
  resnet18_layer4  — ResNet18, обучаются последний блок layer4 и последний слой
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

MODELS = ["cnn", "resnet18", "resnet18_frozen", "resnet18_layer4"]


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class SimpleCNN(nn.Module):
    """5 блоков «свёртка 3×3 → BN → ReLU, дважды → max-pool», затем
    глобальное усреднение и полносвязный слой. 224×224 → 7×7×256 → 256 → классы."""

    def __init__(self, num_classes, channels=(16, 32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, c_in = [], 3
        for c in channels:
            blocks.append(conv_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build(name, num_classes, pretrained=True):
    """Создать модель по имени. pretrained=False — без загрузки весов ImageNet
    (для загрузки своего чекпоинта)."""
    if name == "cnn":
        return SimpleCNN(num_classes)
    if name in ("resnet18", "resnet18_frozen", "resnet18_layer4"):
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if name != "resnet18":
            for n, p in model.named_parameters():
                p.requires_grad = name == "resnet18_layer4" and n.startswith("layer4.")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Неизвестная модель {name}, есть: {', '.join(MODELS)}")


def train_mode(model):
    """Режим обучения. Замороженные слои BatchNorm остаются в режиме оценки,
    чтобы их статистики ImageNet не менялись."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
            m.eval()


def count_params(model, trainable=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


def save(model, name, path, **extra):
    torch.save({"name": name, "state_dict": model.state_dict(), **extra}, path)


def load(path, num_classes, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build(ckpt["name"], num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
