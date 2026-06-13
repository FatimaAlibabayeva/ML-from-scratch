import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    # vectorize edirik yeniki egerki hemin arrayler beraberdise meselen [1 1 0] ve [0 1 0]
    # cavab olacaq == dide [false true true] ve biz meani tapanda 2/3 olacaq
    return float((y_true == y_pred).mean())


def precision(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    """TP / (TP + FP) — of all predicted positives, what fraction were correct."""
    # true postivie tapiriq her ikisindede ortaq 1 olan
    TP = ((y_true == positive_label) & (y_pred == positive_label)).sum()
    FP = ((y_pred == positive_label) & (y_true != positive_label)).sum()
    return float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0  # error vermir ama at least 0 verirde


def recall(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    """TP / (TP + FN) — of all actual positives, what fraction were found."""
    TP = ((y_true == positive_label) & (y_pred == positive_label)).sum()
    # really 1 idi ama model 0 dedi
    FN = ((y_true == positive_label) & (y_pred != positive_label)).sum()
    return float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    """Harmonic mean of precision and recall: 2*P*R / (P+R)."""
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    return float((2 * p * r) / (p + r)) if (p + r) != 0 else 0.0


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area Under the ROC Curve computed by sweeping the decision threshold.

    Tie-handling strategy: we use np.unique() which de-duplicates thresholds, so
    each distinct probability value from the KNN's {0, 1/k, ..., 1} grid is
    tested exactly once. After collecting all (FPR, TPR) points we sort by FPR
    (ascending) before calling trapezoid — this guarantees correct integration
    even if the unsorted threshold sweep produces a non-monotone FPR sequence.
    The curve is anchored at (0,0) by prepending that point explicitly.

    basa dusmeyimcin:
    example y_true=[1,0,1]; y_score=[0.9,0.8,0.1] y_score>=0.6
    then [true true false] olur ve
    biz bunun ucun y_pred yaziriq [1 1 0]
    indi ise bunu y_true ile compare eliyrik [TP FP FN]
    soramdaki TPR=1/1
    FPR=1/2
    ve netice olaraq bize rocun altindaki saheni yeni aucu verecek
    """
    thresholds = np.unique(y_score)[::-1]  # descending — strictest first
    tprs = [0.0]
    fprs = [0.0]
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tpr = recall(y_true, y_pred, positive_label=1)
        # positive diyir pred ama eslinde negativdi (0 di)
        FP = ((y_true != 1) & (y_pred == 1)).sum()
        TN = ((y_true != 1) & (y_pred != 1)).sum()
        fpr = float(FP / (TN + FP)) if (TN + FP) != 0 else 0.0

        tprs.append(tpr)
        fprs.append(fpr)

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # sort by FPR so trapezoid integrates left-to-right correctly
    sort_idx = np.argsort(fprs)
    fprs = fprs[sort_idx]
    tprs = tprs[sort_idx]

    return float(np.trapezoid(tprs, fprs))
