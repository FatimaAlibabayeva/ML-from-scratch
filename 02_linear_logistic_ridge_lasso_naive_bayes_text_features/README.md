# AI-ENG-201 HW2 — Linear Models, Regularization, and Classification

**Author:** Fatima Alibabayeva  
**Course:** AI-ENG-201 Machine Learning, AI Academy  
**Semester:** Fall 2026  

## Structure

```
hw2/
├── report.pdf                  # Report (≤12 pages)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── pledge.txt                  # Signed honor pledge
├── src/
│   ├── linear_regression.py    # Part 1.1 — OLS (normal equations)
│   ├── linear_regression_gd.py # Part 1.2 — Gradient descent regression
│   ├── ridge_regression.py     # Part 2.1 — Ridge (closed-form)
│   ├── lasso_regression.py     # Part 2.2 — Lasso (coordinate descent)
│   ├── logistic_regression.py  # Part 3.1/3.2 — Logistic regression
│   ├── naive_bayes.py          # Part 3.3 — Gaussian Naive Bayes
│   └── text_features.py        # Part 3.4 — BagOfWords, TfidfTransformer
├── notebooks/
│   └── hw2_analysis.ipynb      # Full analysis notebook
├── tests/
│   └── test_linear_models.py   # pytest test suite
└── figures/                    # Generated plots (after running notebook)
```

## How to run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the analysis notebook
```bash
cd hw2/
jupyter notebook notebooks/hw2_analysis.ipynb
```
Run all cells top-to-bottom. All figures will be saved to `figures/`.

### Run tests
```bash
cd hw2/
pytest tests/test_linear_models.py -v
```

## Reproducibility

All random operations use `np.random.seed(42)` (or explicitly stated seeds).
Results reproduce on a fresh kernel with the specified package versions.

## Seeds used

| Experiment | Seed |
|---|---|
| GD convergence plot (N=1000, p=10) | 42 |
| Polynomial fit train/val split | 42 (random_state=42) |
| Wine dataset split | 42 (random_state=42) |
| Logistic Regression sanity check | 42 |
| KFold hyperparameter search | 42 (random_state=42) |
| Random search lambdas | 42 (RandomState(42)) |

## Implementation notes

- All `src/` modules avoid Python loops over training samples — matrix operations only.
- Lasso: features standardized inside `fit()`; `predict()` undoes standardization.
- GaussianNaiveBayes: log-space computation with ε=1e-9 added to variances.
- `text_features.py` does **not** import `sklearn.feature_extraction`.
- `RidgeRegression`: intercept excluded from L2 penalty (I[0,0] = 0).
- `LogisticRegression`: intercept excluded from L2 penalty (reg[0] = 0).
