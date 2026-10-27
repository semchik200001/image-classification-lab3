"""Grad-CAM: на какие области изображения опирается ResNet18 при ответе.

Карта считается по выходу последнего блока layer4 (7×7) для предсказанного
класса и растягивается до размера входа. Для каждого изображения рисуются
карты нескольких моделей рядом.

    python3 gradcam.py data/test/glass/glass_03.jpg data/test/paper/paper_01.jpg
    python3 gradcam.py <файлы> --models resnet18 resnet18_frozen --out results/gradcam.jpg
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw

import models
from data import CLASSES, IMAGE_SIZE, MEAN, STD, make_transform, open_rgb
from evaluate import font


def grad_cam(model, x):
    """Карта значимости 224×224 в [0, 1], предсказанный класс и его вероятность."""
    store = {}

    def hook(m, i, o):
        store["act"] = o
        o.register_hook(lambda g: store.update(grad=g))

    h = model.layer4.register_forward_hook(hook)
    try:
        model.zero_grad()
        out = model(x.clone().requires_grad_(True))
        cls = int(out.argmax(1))
        out[0, cls].backward()
    finally:
        h.remove()
    weights = store["grad"].mean((2, 3), keepdim=True)
    cam = torch.relu((weights * store["act"]).sum(1))[0]
    cam = torch.nn.functional.interpolate(cam[None, None], size=(IMAGE_SIZE, IMAGE_SIZE),
                                          mode="bilinear", align_corners=False)[0, 0]
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()
    return cam, cls, float(torch.softmax(out.detach(), 1)[0, cls])


def overlay(x, cam):
    img = x[0].permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
    heat = colormaps["jet"](cam)[..., :3]
    mix = 0.55 * np.clip(img, 0, 1) + 0.45 * heat
    return Image.fromarray((mix * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM для ResNet18")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--models", nargs="+", help="по умолчанию те из resnet18_frozen, "
                    "resnet18_layer4, resnet18, что есть в models/")
    ap.add_argument("--out", default="results/gradcam.jpg")
    args = ap.parse_args()
    args.models = args.models or [m for m in ("resnet18_frozen", "resnet18_layer4", "resnet18")
                                  if Path(f"models/{m}.pt").exists()]

    nets = [models.load(f"models/{m}.pt", len(CLASSES))[0] for m in args.models]
    tf = make_transform(False)
    s, cap = IMAGE_SIZE, 40
    sheet = Image.new("RGB", ((1 + len(nets)) * s, len(args.images) * (s + cap)), "white")
    d = ImageDraw.Draw(sheet)
    f = font(15)
    for r, path in enumerate(args.images):
        p = Path(path)
        x = tf(open_rgb(p)).unsqueeze(0)
        y = r * (s + cap)
        orig = x[0].permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
        sheet.paste(Image.fromarray((np.clip(orig, 0, 1) * 255).astype(np.uint8)), (0, y))
        d.text((4, y + s + 2), f"{p.stem}", fill="black", font=f)
        d.text((4, y + s + 20), f"верно: {p.parent.name}", fill="black", font=f)
        for c, (name, net) in enumerate(zip(args.models, nets), 1):
            cam, cls, prob = grad_cam(net, x)
            sheet.paste(overlay(x, cam), (c * s, y))
            ok = CLASSES[cls] == p.parent.name
            d.text((c * s + 4, y + s + 2), name, fill="black", font=f)
            d.text((c * s + 4, y + s + 20), f"{CLASSES[cls]} {prob:.2f}",
                   fill=(0, 140, 0) if ok else (200, 0, 0), font=f)
    sheet.save(args.out, quality=88)
    print("Сохранено:", args.out)


if __name__ == "__main__":
    main()
