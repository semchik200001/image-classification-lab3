"""Обучение модели на TrashNet.

Датасет делится на train/val (80/20, стратифицированно, seed 42). Модель
обучается фиксированное число эпох, сохраняются веса после последней эпохи,
лучшая эпоха по val не выбирается.

    python3 train.py --model resnet18
    python3 train.py --model cnn --no-aug --name cnn_noaug

На выходе:
  models/<имя>.pt               — веса и история обучения
  results/history_<имя>.csv     — loss и accuracy по эпохам
  results/split.csv             — какие файлы попали в train и val
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import models
from data import CLASSES, list_images, make_loader, save_split, split
from metrics import confusion, report

DEFAULTS = {"cnn": dict(epochs=100, lr=1e-3),
            "resnet18_frozen": dict(epochs=15, lr=1e-3),
            "resnet18_layer4": dict(epochs=15, lr=3e-4),
            "resnet18": dict(epochs=15, lr=3e-4)}


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, dev):
    """Логиты и истинные метки для всего набора."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(dev)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def eval_loader(model, loader, dev):
    """loss, accuracy и macro-F1 на наборе."""
    logits, labels = predict(model, loader, dev)
    loss = nn.functional.cross_entropy(logits, labels).item()
    rep = report(confusion(labels.numpy(), logits.argmax(1).numpy(), len(CLASSES)))
    return loss, float(rep["accuracy"]), float(rep["macro_f1"])


def train_epoch(model, loader, opt, loss_fn, dev):
    models.train_mode(model)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser(description="Обучение классификатора мусора")
    ap.add_argument("--model", choices=models.MODELS, default="resnet18")
    ap.add_argument("--name", help="имя запуска (по умолчанию как у модели)")
    ap.add_argument("--data", default="data/trashnet")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-aug", action="store_true", help="обучать без аугментаций")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    name = args.name or args.model
    epochs = args.epochs if args.epochs is not None else DEFAULTS[args.model]["epochs"]
    lr = args.lr if args.lr is not None else DEFAULTS[args.model]["lr"]

    seed_all(args.seed)
    dev = device()
    train_s, val_s = split(list_images(args.data), seed=42)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    save_split(train_s, val_s, "results/split.csv")
    train_dl = make_loader(train_s, True, not args.no_aug, args.batch, args.workers)
    val_dl = make_loader(val_s, False, batch_size=64, workers=args.workers)

    model = models.build(args.model, len(CLASSES)).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    print(f"{name}: {args.model}, {dev}, train {len(train_s)}, val {len(val_s)}, "
          f"параметров {models.count_params(model):,} (обучаемых "
          f"{models.count_params(model, trainable=True):,}), эпох {epochs}, lr {lr}, "
          f"аугментации {'нет' if args.no_aug else 'да'}")

    history = []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, loss_fn, dev)
        sched.step()
        val_loss, val_acc, val_f1 = eval_loader(model, val_dl, dev)
        dt = time.perf_counter() - t0
        history.append(dict(epoch=epoch, train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                            val_loss=round(val_loss, 4), val_acc=round(val_acc, 4),
                            val_f1=round(val_f1, 4), time_s=round(dt, 1)))
        print(f"эпоха {epoch:2d}/{epochs}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} F1 {val_f1:.3f} | {dt:.0f} с", flush=True)

    total = time.perf_counter() - t_start
    with open(f"results/history_{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=history[0].keys())
        wr.writeheader()
        wr.writerows(history)
    models.save(model, args.model, f"models/{name}.pt", run=name, classes=CLASSES,
                augment=not args.no_aug, epochs=epochs, lr=lr, train_time_s=round(total, 1),
                history=history)
    print(f"Готово за {total / 60:.1f} мин, веса: models/{name}.pt")


if __name__ == "__main__":
    main()
