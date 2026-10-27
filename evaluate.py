"""Оценка всех обученных моделей и подготовка материалов для отчёта.

Для каждой модели из models/ считаются метрики на train (без аугментаций),
val (20% TrashNet) и собственной тестовой выборке data/test.
В results/ сохраняются:
  summary.csv              — сводная таблица по моделям
  per_class.csv            — precision / recall / F1 по классам для всех моделей на val и test
  background.csv           — accuracy на test по типу фона (разметка в data/test/sources.csv)
  mcnemar.csv              — парное сравнение моделей на test (точный тест Макнемара,
                             p-value с поправкой Холма на число сравнений)
  test_predictions.csv     — предсказания всех моделей на каждом тестовом изображении
  curves.png               — loss и accuracy по эпохам на train и val
  confusion_<набор>.png    — матрицы ошибок лучшей модели на val и test
  confusion_test_all.png   — матрицы ошибок всех моделей на test
  test_grid_<модель>.jpg   — тестовые изображения с предсказаниями каждой модели
  errors_val.jpg           — ошибки лучшей модели на val

    python3 evaluate.py
"""

import csv
import time
from itertools import combinations
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

import models
from data import (CLASSES, DUP_DIST, list_images, make_loader, make_transform,
                  nearest_hash_dist, split)
from metrics import confusion, report
from train import device, predict

RUNS = ["cnn_noaug", "cnn", "resnet18_frozen", "resnet18_layer4", "resnet18"]
TITLES = {"cnn_noaug": "CNN без аугментаций", "cnn": "CNN с аугментациями",
          "resnet18_frozen": "ResNet18, обучен только fc",
          "resnet18_layer4": "ResNet18, обучены layer4 и fc", "resnet18": "ResNet18, дообучение целиком"}
OUT = Path("results")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def wilson(k, n, z=1.96):
    """95%-доверительный интервал Уилсона для доли k из n."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar(ok_a, ok_b):
    """Точный тест Макнемара: сколько изображений верно только у A, только у B, и p-value."""
    a = int((ok_a & ~ok_b).sum())
    b = int((ok_b & ~ok_a).sum())
    n = a + b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0
    return a, b, p


def holm(pvalues):
    """Поправка Холма на множественные сравнения."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def check_split(train_s, val_s, path=OUT / "split.csv"):
    """Разбиение, которое считает evaluate.py, должно совпадать с сохранённым при обучении."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        saved = {(r["class"], r["file"]): r["subset"] for r in csv.DictReader(f)}
    now = {(CLASSES[lbl], p.name): k for k, items in (("train", train_s), ("val", val_s))
           for p, lbl in items}
    if saved != now:
        raise SystemExit(f"Разбиение train/val не совпадает с {path}")


@torch.no_grad()
def time_per_image(model, dev, n=50):
    x = make_transform(False)(Image.new("RGB", (512, 384))).unsqueeze(0).to(dev)
    for _ in range(5):
        model(x)
    if dev.type == "mps":
        torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x).cpu()
    return (time.perf_counter() - t) / n * 1000


def plot_curves(runs):
    fig, axes = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 6.5), squeeze=False)
    for j, (run, ckpt) in enumerate(runs.items()):
        h = ckpt["history"]
        ep = [r["epoch"] for r in h]
        ax = axes[0, j]
        ax.plot(ep, [r["train_loss"] for r in h], label="train")
        ax.plot(ep, [r["val_loss"] for r in h], label="val")
        ax.set_title(TITLES[run], fontsize=10)
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
        ax = axes[1, j]
        ax.plot(ep, [r["train_acc"] for r in h], label="train")
        ax.plot(ep, [r["val_acc"] for r in h], label="val")
        ax.set_ylim(0.2, 1.01)
        ax.set_xlabel("эпоха")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=120)
    plt.close(fig)


def plot_confusion(cm, title, path=None, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("предсказано")
    ax.set_ylabel("на самом деле")
    ax.set_title(title, fontsize=10)
    if fig:
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def grid(items, path, cols=8, size=180):
    """Сетка миниатюр: (путь, истинный класс, предсказанный класс, уверенность).
    Зелёная рамка — верно, красная — ошибка."""
    rows = (len(items) + cols - 1) // cols
    cap = 34
    sheet = Image.new("RGB", (cols * size, rows * (size + cap)), "white")
    d = ImageDraw.Draw(sheet)
    f = font(13)
    for k, (path_, true, pred, conf) in enumerate(items):
        x, y = (k % cols) * size, (k // cols) * (size + cap)
        with Image.open(path_) as im:
            im = ImageOps.fit(im.convert("RGB"), (size - 8, size - 8))
        ok = true == pred
        d.rectangle([x + 1, y + 1, x + size - 2, y + size - 2], outline=(0, 160, 0) if ok else (220, 0, 0), width=4)
        sheet.paste(im, (x + 4, y + 4))
        d.text((x + 4, y + size), f"{CLASSES[pred]} {conf:.2f}", fill="black", font=f)
        if not ok:
            d.text((x + 4, y + size + 15), f"верно: {CLASSES[true]}", fill=(200, 0, 0), font=f)
    sheet.save(path, quality=88)


def main():
    dev = device()
    train_s, val_s = split(list_images("data/trashnet"))
    check_split(train_s, val_s)
    test_s = list_images("data/test")
    val_dist = nearest_hash_dist([p for p, _ in val_s], [p for p, _ in train_s])
    val_clean = val_dist > DUP_DIST
    print(f"val: {(~val_clean).sum()} изображений почти совпадают с train "
          f"(расстояние хеша <= {DUP_DIST})")
    sets = {"train": train_s, "val": val_s, "test": test_s}
    loaders = {k: make_loader(v, False, batch_size=64) for k, v in sets.items()}
    counts = {k: np.bincount([lbl for _, lbl in v], minlength=len(CLASSES)) for k, v in sets.items()}
    for k, c in counts.items():
        print(f"{k}: {len(sets[k])} изображений, по классам " +
              ", ".join(f"{cls} {n}" for cls, n in zip(CLASSES, c)))

    rows, ckpts, preds = [], {}, {}
    for run in RUNS:
        path = Path("models") / f"{run}.pt"
        if not path.exists():
            continue
        model, ckpt = models.load(path, len(CLASSES), dev)
        if ckpt.get("classes", CLASSES) != CLASSES:
            raise SystemExit(f"{path}: другой список классов {ckpt['classes']}")
        ckpts[run] = ckpt
        row = dict(run=run, params=models.count_params(model), epochs=ckpt["epochs"],
                   train_min=round(ckpt["train_time_s"] / 60, 1))
        for k, dl in loaders.items():
            logits, labels = predict(model, dl, dev)
            probs = torch.softmax(logits, 1).numpy()
            preds[run, k] = (probs, labels.numpy())
            rep = report(confusion(labels.numpy(), probs.argmax(1), len(CLASSES)))
            row[f"{k}_acc"] = round(rep["accuracy"], 4)
            row[f"{k}_f1"] = round(rep["macro_f1"], 4)
            row[f"{k}_loss"] = round(torch.nn.functional.cross_entropy(logits, labels).item(), 4)
        probs, labels = preds[run, "val"]
        row["val_acc_nodup"] = round(float((probs.argmax(1) == labels)[val_clean].mean()), 4)
        probs, labels = preds[run, "test"]
        lo, hi = wilson(int((probs.argmax(1) == labels).sum()), len(labels))
        row["test_acc_ci"] = f"{lo:.3f}-{hi:.3f}"
        row["ms_per_image"] = round(time_per_image(model, dev), 1)
        rows.append(row)
        print(f"{run}: train {row['train_acc']:.3f} | val acc {row['val_acc']:.3f} "
              f"F1 {row['val_f1']:.3f} | test acc {row['test_acc']:.3f} [{row['test_acc_ci']}] "
              f"F1 {row['test_f1']:.3f} | "
              f"{row['ms_per_image']} мс/изобр.")

    if not rows:
        raise SystemExit("В models/ нет обученных моделей")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    plot_curves(ckpts)

    best = max(rows, key=lambda r: r["val_f1"])["run"]
    print("Лучшая по F1 на val:", best)
    with open(OUT / "per_class.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run", "set", "class", "precision", "recall", "f1", "support"])
        for run in ckpts:
            for k in ("val", "test"):
                probs, labels = preds[run, k]
                cm = confusion(labels, probs.argmax(1), len(CLASSES))
                rep = report(cm)
                for i, cls in enumerate(CLASSES):
                    wr.writerow([run, k, cls, round(rep["precision"][i], 3), round(rep["recall"][i], 3),
                                 round(rep["f1"][i], 3), int(rep["support"][i])])
                if run == best:
                    plot_confusion(cm, f"{TITLES[best]}, {k}: accuracy {rep['accuracy']:.3f}",
                                   OUT / f"confusion_{k}.png")

    with open(OUT / "test_predictions.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "class", *[f"{r}_pred" for r in ckpts], *[f"{r}_conf" for r in ckpts]])
        for i, (p, lbl) in enumerate(test_s):
            pr = [preds[r, "test"][0][i] for r in ckpts]
            wr.writerow([f"{p.parent.name}/{p.name}", CLASSES[lbl], *[CLASSES[x.argmax()] for x in pr],
                         *[round(float(x.max()), 3) for x in pr]])

    with open("data/test/sources.csv", encoding="utf-8") as f:
        bg_map = {(r["class"], r["file"]): r["background"] for r in csv.DictReader(f)}
    bg = np.array([bg_map[p.parent.name, p.name] for p, _ in test_s])
    with open(OUT / "background.csv", "w", newline="") as f:
        wr = csv.writer(f)
        kinds = ["light", "closeup", "scene"]
        wr.writerow(["run", *[f"{k} ({(bg == k).sum()})" for k in kinds]])
        for r in ckpts:
            probs, labels = preds[r, "test"]
            ok = probs.argmax(1) == labels
            acc = [round(float(ok[bg == k].mean()), 3) for k in kinds]
            wr.writerow([r, *acc])
            print(f"{r} на test по фону: " + ", ".join(f"{k} {a:.3f}" for k, a in zip(kinds, acc)))

    ok = {r: preds[r, "test"][0].argmax(1) == preds[r, "test"][1] for r in ckpts}
    with open(OUT / "mcnemar.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["a", "b", "only_a_correct", "only_b_correct", "p_value", "p_holm"])
        tests = [(ra, rb, *mcnemar(ok[ra], ok[rb])) for ra, rb in combinations(ckpts, 2)]
        adj = holm(np.array([t[4] for t in tests]))
        for t, pa in zip(tests, adj):
            wr.writerow([*t[:4], round(t[4], 4), round(float(pa), 4)])

    cols = 3
    rows_n = (len(ckpts) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.8 * cols, 4.6 * rows_n), squeeze=False)
    for ax in axes.ravel()[len(ckpts):]:
        ax.axis("off")
    for ax, r in zip(axes.ravel(), ckpts):
        probs, labels = preds[r, "test"]
        cm = confusion(labels, probs.argmax(1), len(CLASSES))
        plot_confusion(cm, f"{TITLES[r]}\naccuracy {report(cm)['accuracy']:.3f}", ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_test_all.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    for r in ckpts:
        probs, labels = preds[r, "test"]
        grid([(p, lbl, int(probs[i].argmax()), float(probs[i].max())) for i, (p, lbl) in enumerate(test_s)],
             OUT / f"test_grid_{r}.jpg", cols=12, size=150)
    probs, labels = preds[best, "val"]
    wrong = [(p, lbl, int(probs[i].argmax()), float(probs[i].max()))
             for i, (p, lbl) in enumerate(val_s) if probs[i].argmax() != lbl]
    grid(wrong[:32], OUT / "errors_val.jpg", cols=8, size=150)


if __name__ == "__main__":
    main()
