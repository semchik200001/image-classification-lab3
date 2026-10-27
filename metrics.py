"""Метрики качества классификации по матрице ошибок."""

import numpy as np


def confusion(y_true, y_pred, num_classes):
    """cm[i, j] — сколько изображений класса i предсказано как j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def report(cm):
    """accuracy, macro-F1 и precision / recall / F1 по каждому классу.
    Если у класса нет предсказаний или примеров, соответствующая метрика равна 0."""
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(0), cm.sum(1)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    s = precision + recall
    f1 = np.divide(2 * precision * recall, s, out=np.zeros_like(tp), where=s > 0)
    return dict(accuracy=tp.sum() / cm.sum(), macro_f1=f1.mean(),
                precision=precision, recall=recall, f1=f1, support=true)
