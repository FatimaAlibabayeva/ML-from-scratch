"""
experiments.py - runs the HW1 experiments and regenerates every report figure.
Run from the hw1/ root: python src/experiments.py
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.knn import ApproxKNN, KNN
from src.metrics import accuracy, f1_score, precision, recall, roc_auc
from src.splits import stratified_kfold, stratified_split

np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "titanic3.csv")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
COLS = [
    "pclass", "sex", "age", "sibsp", "parch", "fare",
    "embarked_C", "embarked_Q", "embarked_S",
]
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]


def fit_preprocessor(train_df: pd.DataFrame) -> dict:
    """Fit imputation values using only the training partition."""
    # bilrikki mean tesirlenir qiymetderden ona gore bos xanalara median yaziriq
    # P.S burada median/mode ancaq train partition-dan fit olunur ki leakage olmasin
    return {
        "age_median": float(train_df["age"].median()),
        "fare_median": float(train_df["fare"].median()),
        "embarked_mode": train_df["embarked"].mode()[0],
    }


def transform_features(df_part: pd.DataFrame, prep: dict) -> np.ndarray:
    """Apply train-fitted imputation and encoding, returning the final feature matrix."""
    # encoding elyri cunki bunnarin deyerleri meselen female ve male reqem deyil
    # ama KNN ancaq reqemnen distance hesablayib isliye bilir
    df = df_part[FEATURES].copy()
    df["age"] = df["age"].fillna(prep["age_median"])
    df["fare"] = df["fare"].fillna(prep["fare_median"])
    df["embarked"] = df["embarked"].fillna(prep["embarked_mode"])
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df = pd.get_dummies(df, columns=["embarked"], drop_first=False)
    for col in ["embarked_C", "embarked_Q", "embarked_S"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].astype(int)
    return df[COLS].astype(float).values


def make_eda_figures(raw_df: pd.DataFrame) -> None:
    """Generate the EDA figures referenced in the report."""
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = raw_df["survived"].value_counts().sort_index()
    ax.bar(["Did not survive (0)", "Survived (1)"], counts.values)
    ax.set_xlabel("Survival Status")
    ax.set_ylabel("Count")
    ax.set_title("Target Distribution: survived")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "target_dist.pdf"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ct = pd.crosstab(raw_df["sex"], raw_df["survived"]).loc[["female", "male"]]
    x = np.arange(len(ct.index))
    width = 0.35
    ax.bar(x - width / 2, ct[0].values, width, label="No")
    ax.bar(x + width / 2, ct[1].values, width, label="Yes")
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index)
    ax.set_xlabel("Sex")
    ax.set_ylabel("Count")
    ax.set_title("Survival by Sex")
    ax.legend(title="Survived")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sex_survival.pdf"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    age0 = raw_df.loc[raw_df["survived"] == 0, "age"].dropna()
    age1 = raw_df.loc[raw_df["survived"] == 1, "age"].dropna()
    ax.boxplot([age0, age1], tick_labels=["No (0)", "Yes (1)"])
    ax.set_xlabel("Survived")
    ax.set_ylabel("Age")
    ax.set_title("Age Distribution by Survival")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "age_survival.pdf"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    fare0 = raw_df.loc[raw_df["survived"] == 0, "fare"].dropna()
    fare1 = raw_df.loc[raw_df["survived"] == 1, "fare"].dropna()
    ax.boxplot([fare0, fare1], tick_labels=["No (0)", "Yes (1)"])
    ax.set_xlabel("Survived")
    ax.set_ylabel("Fare")
    ax.set_title("Fare Distribution by Survival")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fare_survival.pdf"), dpi=300)
    plt.close()


def metrics_dict(y_true, y_pred, y_score=None):
    result = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_score is not None:
        result["roc_auc"] = roc_auc(y_true, y_score)
    return result


# ------------------------------------------------------------------
# 0. Data loading, EDA figures, and honest preprocessing
# ------------------------------------------------------------------
# 0.  Data loading and cleaning
# ------------------------------------------------------------------
# burada split-i indexler uzerinden edirik ki raw dataframe qalir, sonra imputation/encoding edilir
# ------------------------------------------------------------------
raw = pd.read_csv(DATA_PATH)
make_eda_figures(raw)

idx = np.arange(len(raw)).reshape(-1, 1)
y_all = raw["survived"].values.astype(int)
idx_train, idx_val, idx_test, y_train, y_val, y_test = stratified_split(idx, y_all, seed=42)
idx_train = idx_train.ravel().astype(int)
idx_val = idx_val.ravel().astype(int)
idx_test = idx_test.ravel().astype(int)

train_raw = raw.iloc[idx_train].reset_index(drop=True)
val_raw = raw.iloc[idx_val].reset_index(drop=True)
test_raw = raw.iloc[idx_test].reset_index(drop=True)

prep = fit_preprocessor(train_raw)
X_train = transform_features(train_raw, prep)
X_val = transform_features(val_raw, prep)
X_test = transform_features(test_raw, prep)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Train-fitted imputation values: {prep}")

# ------------------------------------------------------------------
# 1. Part 2.5 - Computational benchmark
# ------------------------------------------------------------------
N_VALUES = [100, 500, 1000, 5000, 10000]
bench_times = []
rng = np.random.default_rng(42)
Xt_fixed = rng.random((100, 9))  # fixed test set required by the assignment; loop-dan colde qalmalidir
for N in N_VALUES:
    Xtr = rng.random((N, 9))  # synthetic train datasi: N artdiqca vaxtin nece artdigina baxiriq
    ytr = rng.integers(0, 2, N)
    knn = KNN(k=5).fit(Xtr, ytr)
    start = time.perf_counter()
    knn.predict(Xt_fixed)
    bench_times.append(time.perf_counter() - start)

# empirical scaling exponent via log-log linear fit
exponent = np.polyfit(np.log(N_VALUES), np.log(bench_times), 1)[0]
print(f"Empirical scaling exponent: {exponent:.3f}  (theoretical is about 1.0 for fixed p)")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(N_VALUES, bench_times, marker="o")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Training-set size N")
ax.set_ylabel("Prediction time (seconds)")
ax.set_title(f"KNN Prediction Time vs N (empirical exponent {exponent:.2f})")
ax.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "benchmark.pdf"), dpi=300)
plt.close()

# ------------------------------------------------------------------
# 2. Part 3.3 - Validation sweep and train-vs-validation figure
# ------------------------------------------------------------------
# 2.  Part 3.3 — Hyperparameter sweep on validation set  (k_sweep.pdf)
# ------------------------------------------------------------------
# k-lari yoxlayiriq, amma test set-e toxunmuruq; best k ancaq validation F1 ile secilir
# ------------------------------------------------------------------
sweep_results = []
for k in K_VALUES:
    model = KNN(k=k, metric="euclidean").fit(X_train, y_train)  # her k ucun modeli train setde saxlayiriq
    y_pred_val = model.predict(X_val)  # validation prediction; k secimi bununla edilir
    y_score_val = model.predict_proba(X_val)[:, 1]  # AUC ucun survived=1 probability lazimdir
    y_pred_train = model.predict(X_train)  # bias-variance ucun train F1 de saxlayiriq
    sweep_results.append({
        "k": k,
        "train_f1": f1_score(y_train, y_pred_train),
        "accuracy": accuracy(y_val, y_pred_val),
        "precision": precision(y_val, y_pred_val),
        "recall": recall(y_val, y_pred_val),
        "f1": f1_score(y_val, y_pred_val),
        "roc_auc": roc_auc(y_val, y_score_val),
    })
sweep_df = pd.DataFrame(sweep_results)
best_k = int(sweep_df.loc[sweep_df["f1"].idxmax(), "k"])
print("\nValidation sweep:\n", sweep_df.to_string(index=False))
print(f"Best k by validation F1: {best_k}")

metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, metric in zip(axes, metrics_list):
    ax.plot(sweep_df["k"], sweep_df[metric], marker="o")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric} vs k")
    ax.grid(True, ls="--", alpha=0.4)
plt.suptitle("Validation-set metric sweep over k", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "k_sweep.pdf"), dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(sweep_df["k"], sweep_df["train_f1"], marker="o", label="Training F1")
ax.plot(sweep_df["k"], sweep_df["f1"], marker="s", label="Validation F1")
ax.set_xlabel("k (number of neighbors)")
ax.set_ylabel("F1 score")
ax.set_title("Training vs Validation F1 across k")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "train_vs_val_f1.pdf"), dpi=300)
plt.close()

# ------------------------------------------------------------------
# 3. Part 4.2 - CV F1 vs k with preprocessing fit inside each fold
# ------------------------------------------------------------------
# 3.  Part 4.2 — CV F1 vs k  (cv_f1_vs_k.pdf)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
trainval_raw = pd.concat([train_raw, val_raw], ignore_index=True)
# training+validation combined = everything except held-out test
y_trainval = np.concatenate([y_train, y_val])
trainval_idx = np.arange(len(trainval_raw)).reshape(-1, 1)

cv_results = []
for k in K_VALUES:
    f1s = []
    for tr_idx_arr, va_idx_arr, ytr, yv in stratified_kfold(trainval_idx, y_trainval, K=5):
        tr_idx = tr_idx_arr.ravel().astype(int)
        va_idx = va_idx_arr.ravel().astype(int)
        fold_prep = fit_preprocessor(trainval_raw.iloc[tr_idx])  # fold icinde fit edilir, validation melumati gormur
        Xtr = transform_features(trainval_raw.iloc[tr_idx], fold_prep)
        Xv = transform_features(trainval_raw.iloc[va_idx], fold_prep)
        m = KNN(k=k, metric="euclidean").fit(Xtr, ytr)
        f1s.append(f1_score(yv, m.predict(Xv)))
    cv_results.append({"k": k, "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s))})
cv_df = pd.DataFrame(cv_results)
best_cv_k = int(cv_df.loc[cv_df["mean_f1"].idxmax(), "k"])
print("\nCV results:\n", cv_df.to_string(index=False))
print(f"Best k by CV F1: {best_cv_k}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cv_df["k"], cv_df["mean_f1"], marker="o", label="Mean F1")
ax.fill_between(cv_df["k"], cv_df["mean_f1"] - cv_df["std_f1"], cv_df["mean_f1"] + cv_df["std_f1"], alpha=0.3, label="+/- 1 std")
ax.set_xlabel("k (number of neighbors)")
ax.set_ylabel("F1 Score")
ax.set_title("5-Fold Stratified CV - Mean F1 vs k")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cv_f1_vs_k.pdf"), dpi=300)
plt.close()

# ------------------------------------------------------------------
# 4. Part 4.5 - Scaling experiment, scaler fit on training fold only
# ------------------------------------------------------------------
# 4.  Part 4.5 — Scaling experiment  (scaling_effect.pdf)
# ------------------------------------------------------------------
# F1 in cox azalmaqi underfittinge yol aca biler yk
# ------------------------------------------------------------------
scaled_results = []
for k in K_VALUES:
    f1s = []
    for tr_idx_arr, va_idx_arr, ytr, yv in stratified_kfold(trainval_idx, y_trainval, K=5):
        tr_idx = tr_idx_arr.ravel().astype(int)
        va_idx = va_idx_arr.ravel().astype(int)
        fold_prep = fit_preprocessor(trainval_raw.iloc[tr_idx])  # fold icinde fit edilir, validation melumati gormur
        Xtr = transform_features(trainval_raw.iloc[tr_idx], fold_prep)
        Xv = transform_features(trainval_raw.iloc[va_idx], fold_prep)
        scaler = StandardScaler()
        # scaler fit edilir ancaq training folda — leakage olmamasi ucun
        Xtr_s = scaler.fit_transform(Xtr)
        Xv_s = scaler.transform(Xv)
        m = KNN(k=k, metric="euclidean").fit(Xtr_s, ytr)
        f1s.append(f1_score(yv, m.predict(Xv_s)))
    scaled_results.append({"k": k, "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s))})
scaled_df = pd.DataFrame(scaled_results)
best_scaled_k = int(scaled_df.loc[scaled_df["mean_f1"].idxmax(), "k"])
print("\nScaled CV results:\n", scaled_df.to_string(index=False))
print(f"Best scaled k by CV F1: {best_scaled_k}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cv_df["k"], cv_df["mean_f1"], marker="o", label="Unscaled")
ax.plot(scaled_df["k"], scaled_df["mean_f1"], marker="s", label="Scaled (StandardScaler)")
ax.fill_between(cv_df["k"], cv_df["mean_f1"] - cv_df["std_f1"], cv_df["mean_f1"] + cv_df["std_f1"], alpha=0.15)
ax.fill_between(scaled_df["k"], scaled_df["mean_f1"] - scaled_df["std_f1"], scaled_df["mean_f1"] + scaled_df["std_f1"], alpha=0.15)
ax.set_xlabel("k (number of neighbors)")
ax.set_ylabel("Mean CV F1 Score")
ax.set_title("Effect of Feature Scaling on CV F1 vs k")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "scaling_effect.pdf"), dpi=300)
plt.close()

# ------------------------------------------------------------------
# 5. Part 3.4 - Baselines comparison
# ------------------------------------------------------------------
# 5.  Part 3.4 — Baselines comparison
# ------------------------------------------------------------------
# ------------------------------------------------------------------
model_best = KNN(k=best_k, metric="euclidean").fit(X_train, y_train)
y_pred_my = model_best.predict(X_val)
y_score_my = model_best.predict_proba(X_val)[:, 1]

majority_class = int(pd.Series(y_train).mode()[0])
# en axmaq modelden yeniki baselindan yaxsidirsa demeli at least nese oyrenib
y_majority = np.full(len(y_val), majority_class, dtype=int)  # hamisina majority class de
y_score_majority = np.full(len(y_val), float(majority_class))  # constant score oldugu ucun AUC 0.5 olur

sk_knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean").fit(X_train, y_train)
y_pred_sk = sk_knn.predict(X_val)
y_score_sk = sk_knn.predict_proba(X_val)[:, 1]

baseline_rows = []
for name, yp, ys in [
    ("Majority baseline", y_majority, y_score_majority),
    (f"My KNN (k={best_k})", y_pred_my, y_score_my),
    (f"sklearn KNN (k={best_k})", y_pred_sk, y_score_sk),
]:
    row = {"model": name, **metrics_dict(y_val, yp, ys)}
    baseline_rows.append(row)
baseline_df = pd.DataFrame(baseline_rows)
print("\nBaseline comparison:\n", baseline_df.to_string(index=False))

# ------------------------------------------------------------------
# 6. Part 3.5 - Final test-set evaluation (touched exactly once)
# ------------------------------------------------------------------
# 6.  Part 3.5 — Final test-set evaluation (touched exactly ONCE)
# ------------------------------------------------------------------
# test set sacred-dir: sadece en axirda best k secilenden sonra baxiriq
# ------------------------------------------------------------------
y_pred_test = model_best.predict(X_test)
y_score_test = model_best.predict_proba(X_test)[:, 1]
test_metrics = metrics_dict(y_test, y_pred_test, y_score_test)
print(f"\nFinal TEST SET Evaluation (k={best_k}):")
for key, value in test_metrics.items():
    print(f"{key}: {value:.4f}")

# ------------------------------------------------------------------
# 7. Optional bonus note
# ------------------------------------------------------------------
# ApproxKNN is implemented in src/knn.py. The full N=100000 BallTree
# benchmark is intentionally not executed by default because it is slow
# on small grading machines; the core required figures above are fully
# reproducible with this script.

print("\nAll experiments done. Figures saved to figures/.")
