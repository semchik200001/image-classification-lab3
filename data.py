"""Загрузка изображений, разбиение train/val и преобразования.

Датасет хранится как папка с подпапками-классами:
  data/trashnet/cardboard/*.jpg, data/trashnet/glass/*.jpg, ...
Тестовая выборка устроена так же: data/test/<класс>/*.jpg
"""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASSES_RU = {"cardboard": "картон", "glass": "стекло", "metal": "металл",
              "paper": "бумага", "plastic": "пластик", "trash": "мусор"}
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png"}
DUP_DIST = 10


def list_images(root):
    """Список (путь, номер класса) по подпапкам-классам, в фиксированном порядке."""
    root = Path(root)
    samples = []
    for label, cls in enumerate(CLASSES):
        if not (root / cls).is_dir():
            raise FileNotFoundError(f"Нет папки класса {root / cls}")
        files = sorted(p for p in (root / cls).iterdir()
                       if p.suffix.lower() in EXTS and not p.name.startswith("."))
        samples += [(p, label) for p in files]
    if not samples:
        raise ValueError(f"В {root} нет изображений")
    return samples


def open_rgb(path):
    """Изображение в RGB с учётом поворота из EXIF."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def average_hash(path, size=16):
    """Перцептивный хеш size×size бит: пиксель уменьшенной копии ярче среднего или нет.
    У копии изображения (уменьшенной, пережатой) расстояние Хэмминга до оригинала близко к 0."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size)), dtype=np.float32)
    return (a > a.mean()).ravel()


def nearest_hash_dist(paths, ref_paths):
    """Для каждого изображения из paths — расстояние Хэмминга до ближайшего из ref_paths."""
    ref = np.stack([average_hash(p) for p in ref_paths])
    return np.array([(ref != average_hash(p)).sum(1).min() for p in paths])


def split(samples, val_frac=0.2, seed=42):
    """Стратифицированное разбиение: в каждом классе доля val_frac уходит в val."""
    rng = random.Random(seed)
    train, val = [], []
    for label in range(len(CLASSES)):
        items = [s for s in samples if s[1] == label]
        rng.shuffle(items)
        n_val = round(len(items) * val_frac)
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def save_split(train, val, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "class", "subset"])
        for subset, items in (("train", train), ("val", val)):
            for p, label in items:
                wr.writerow([Path(p).name, CLASSES[label], subset])


def make_transform(train, augment=True, size=IMAGE_SIZE):
    """Обучение: случайный кроп, отражения, поворот, яркость и цвет.
    Проверка: уменьшение по короткой стороне и центральный кроп."""
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if train and augment:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            *norm])
    return T.Compose([T.Resize(size), T.CenterCrop(size), *norm])


class ImageList(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(open_rgb(path)), label


def make_loader(samples, train, augment=True, batch_size=32, workers=4):
    ds = ImageList(samples, make_transform(train, augment))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      persistent_workers=workers > 0)
