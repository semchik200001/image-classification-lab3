"""Тесты: разбиение данных, метрики, модели, сохранение весов, тестовая выборка."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import models
from data import (CLASSES, DUP_DIST, ImageList, list_images, make_transform,
                  nearest_hash_dist, split)
from evaluate import holm, mcnemar, wilson
from metrics import confusion, report

TRASHNET = Path("data/trashnet")
TEST = Path("data/test")


def fake_samples(counts):
    return [(Path(f"{CLASSES[c]}_{i}.jpg"), c) for c, n in enumerate(counts) for i in range(n)]


def test_split_stratified_and_deterministic():
    samples = fake_samples([403, 501, 410, 594, 482, 137])
    train, val = split(samples, val_frac=0.2, seed=42)
    assert len(train) + len(val) == len(samples)
    assert not {p for p, _ in train} & {p for p, _ in val}
    for c, n in enumerate([403, 501, 410, 594, 482, 137]):
        assert sum(1 for _, lbl in val if lbl == c) == round(n * 0.2)
    assert split(samples, seed=42) == (train, val)
    assert split(samples, seed=1) != (train, val)


def test_confusion_and_report():
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    cm = confusion(y_true, y_pred, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    rep = report(cm)
    assert rep["accuracy"] == pytest.approx(4 / 6)
    assert rep["precision"] == pytest.approx([1, 0.5, 0.5])
    assert rep["recall"] == pytest.approx([2 / 3, 0.5, 1])
    f1 = [0.8, 0.5, 2 / 3]
    assert rep["f1"] == pytest.approx(f1)
    assert rep["macro_f1"] == pytest.approx(np.mean(f1))


def test_report_empty_class():
    rep = report(confusion([0, 0, 1], [0, 0, 0], 3))
    assert rep["f1"][1] == 0 and rep["f1"][2] == 0
    assert np.isfinite(rep["macro_f1"])


def test_wilson():
    lo, hi = wilson(36, 72)
    assert lo == pytest.approx(0.388, abs=1e-3) and hi == pytest.approx(0.612, abs=1e-3)
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0, abs=1e-12) and 0 < hi < 0.35


def test_holm():
    adj = holm(np.array([0.01, 0.04, 0.03, 0.5]))
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_mcnemar():
    a = np.array([True] * 15 + [False] * 5 + [True] * 30)
    b = np.array([False] * 15 + [True] * 5 + [True] * 30)
    assert mcnemar(a, b)[:2] == (15, 5)
    assert mcnemar(a, b)[2] == pytest.approx(0.0414, abs=1e-4)
    assert mcnemar(a, a)[2] == 1.0


def test_transforms_shape(tmp_path):
    for size in [(512, 384), (300, 600), (100, 100)]:
        img = Image.new("RGB", size, (120, 30, 200))
        for train in (True, False):
            x = make_transform(train)(img)
            assert x.shape == (3, 224, 224)


def test_dataset_reads_png_with_alpha(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGBA", (64, 48)).save(p)
    x, y = ImageList([(p, 3)], make_transform(False))[0]
    assert x.shape == (3, 224, 224) and y == 3


@pytest.mark.parametrize("name", models.MODELS)
def test_model_output_shape(name):
    model = models.build(name, len(CLASSES), pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, len(CLASSES))


def test_frozen_trains_only_head():
    model = models.build("resnet18_frozen", len(CLASSES), pretrained=False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["fc.weight", "fc.bias"]
    models.train_mode(model)
    assert model.fc.training
    assert not model.bn1.training


def test_frozen_step_keeps_backbone():
    """Шаг обучения замороженной ResNet18 меняет только fc: веса и статистики BN прежние."""
    torch.manual_seed(0)
    model = models.build("resnet18_frozen", len(CLASSES), pretrained=False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    models.train_mode(model)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 3, 64, 64)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    opt.step()
    after = model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed == {"fc.weight", "fc.bias"}


def test_layer4_trains_last_block():
    model = models.build("resnet18_layer4", len(CLASSES), pretrained=False)
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "fc"}
    models.train_mode(model)
    assert model.layer4[0].bn1.training
    assert not model.layer3[0].bn1.training


def test_cnn_overfits_tiny_batch():
    """Сеть способна запомнить маленький набор: loss падает почти до нуля."""
    torch.manual_seed(0)
    model = models.build("cnn", len(CLASSES))
    x = torch.randn(12, 3, 64, 64)
    y = torch.arange(12) % len(CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


@pytest.mark.parametrize("name", ["cnn", "resnet18_layer4"])
def test_save_load_roundtrip(tmp_path, name):
    model = models.build(name, len(CLASSES), pretrained=False).eval()
    models.save(model, name, tmp_path / "m.pt", epochs=1, history=[dict(epoch=1, val_acc=0.5)])
    loaded, ckpt = models.load(tmp_path / "m.pt", len(CLASSES))
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
    assert ckpt["epochs"] == 1 and ckpt["name"] == name


def test_test_set_complete():
    """Тестовая выборка: 72 изображения, по 12 на класс, все описаны в sources.csv
    с автором, лицензией и типом фона."""
    samples = list_images(TEST)
    assert len(samples) == 72
    assert np.bincount([lbl for _, lbl in samples]).tolist() == [12] * len(CLASSES)
    with open(TEST / "sources.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {(r["class"], r["file"]) for r in rows} == {(p.parent.name, p.name) for p, _ in samples}
    assert all(r["author"] and r["license"] and r["page"] for r in rows)
    assert {r["background"] for r in rows} == {"light", "closeup", "scene"}
    for p, _ in samples:
        with Image.open(p) as im:
            im.verify()


@pytest.mark.skipif(not TRASHNET.exists(), reason="TrashNet не скачан")
def test_test_set_not_in_trashnet():
    """В тестовой выборке нет копий изображений из TrashNet. Сравнение по хешу 16×16
    (256 бит): у уменьшенной и пережатой копии расстояние 0, у самой похожей
    пары разных изображений 16."""
    test = list_images(TEST)
    dist = nearest_hash_dist([p for p, _ in test], [p for p, _ in list_images(TRASHNET)])
    near = [str(p) for (p, _), d in zip(test, dist) if d <= DUP_DIST]
    assert not near, f"похожи на изображения из TrashNet: {near}"
