import numpy as np
from typing import Iterator, Tuple

# Fold type: (X_train, X_val, y_train, y_val)
Fold = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train / validation / test subsets while preserving class proportions
    in each split to within 0.5 percentage points (stratified split).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    np.random.seed(seed)

    # 0a beraber olannari bir yere yigir
    idx_0 = np.where(y == 0)[0]
    # 1e olannari bir yere
    idx_1 = np.where(y == 1)[0]
    # cunki biz bilirkki hem valda hem test hemde trainde data eyni derecede edaltedi olmalidir

    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    # datanin necesinin train olduqunu tapiriq
    n0_train = int(len(idx_0) * train_frac)
    n1_train = int(len(idx_1) * train_frac)

    # ve necesinin validation
    n0_val = int(len(idx_0) * val_frac)
    n1_val = int(len(idx_1) * val_frac)

    # indise birlesidrek (concatenate)
    # concatenate listleri birlesdirir (0larin ve 1lerin indexlerini)
    train_idx = np.concatenate([idx_0[:n0_train], idx_1[:n1_train]])
    val_idx = np.concatenate([idx_0[n0_train : n0_train + n0_val], idx_1[n1_train : n1_train + n1_val]])
    test_idx = np.concatenate([idx_0[n0_train + n0_val :], idx_1[n1_train + n1_val :]])

    # heim indexlere uygun dataalri goturur
    return (
        x[train_idx], x[val_idx], x[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


def stratified_kfold(X: np.ndarray, y: np.ndarray, K: int = 5, seed: int = 42) -> Iterator[Fold]:
    """
    Stratified K-fold cross-validation: yields K folds, each preserving class proportions.

    Yields
    ------
    (X_train_fold, X_val_fold, y_train_fold, y_val_fold) for each fold.
    """
    np.random.seed(seed)

    # where tuple qaytarir ve icindeki arraycin [0] yaziriq
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    # soramsa k hisseye boluruk
    folds_0 = np.array_split(idx_0, K)
    folds_1 = np.array_split(idx_1, K)

    for i in range(K):
        # validationcin secirik
        val_idx = np.concatenate([folds_0[i], folds_1[i]])

        # traincin secirik index
        train_idx = np.concatenate(
            [folds_0[j] for j in range(K) if j != i]
            + [folds_1[j] for j in range(K) if j != i]
        )

        yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]
