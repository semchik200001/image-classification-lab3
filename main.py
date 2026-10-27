"""Классификация изображений обученной моделью.

    python3 main.py photo.jpg
    python3 main.py data/test/glass --model models/cnn.pt --top 3
"""

import argparse
from pathlib import Path

import torch

import models
from data import CLASSES, CLASSES_RU, EXTS, make_transform, open_rgb
from train import device


def classify(model, paths, dev, top=3):
    """Для каждого изображения — список (класс, вероятность) по убыванию вероятности."""
    tf = make_transform(False)
    out = []
    with torch.no_grad():
        for p in paths:
            x = tf(open_rgb(p)).unsqueeze(0).to(dev)
            prob = torch.softmax(model(x), 1)[0].cpu()
            val, idx = prob.topk(min(top, len(CLASSES)))
            out.append([(CLASSES[i], v) for i, v in zip(idx.tolist(), val.tolist())])
    return out


def main():
    ap = argparse.ArgumentParser(description="Классификация мусора по фото")
    ap.add_argument("inputs", nargs="+", help="изображения или папки с изображениями")
    ap.add_argument("--model", default="models/resnet18_layer4.pt")
    ap.add_argument("--top", type=int, default=3, help="сколько классов показывать")
    args = ap.parse_args()

    paths = []
    for s in map(Path, args.inputs):
        if not s.exists():
            raise SystemExit(f"Нет файла {s}")
        paths += sorted(p for p in s.rglob("*") if p.suffix.lower() in EXTS) if s.is_dir() else [s]
    dev = device()
    model, _ = models.load(args.model, len(CLASSES), dev)
    for p, res in zip(paths, classify(model, paths, dev, args.top)):
        print(f"{p}: " + ", ".join(f"{c} ({CLASSES_RU[c]}) {v:.2f}" for c, v in res))


if __name__ == "__main__":
    main()
