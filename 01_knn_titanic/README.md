# AI-ENG-201 HW1 - k-Nearest Neighbors

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
The submission includes `data/titanic3.csv` from the Vanderbilt biostat Titanic dataset.

## Running experiments
From the project root:
```bash
python src/experiments.py
```
This regenerates the report figures in `figures/` and prints the validation, CV, baseline, and final test metrics.

## Running tests
```bash
pytest -q
```
Current result: `14 passed`.

## Directory layout
```
hw1/
├── report.pdf
├── requirements.txt
├── README.md
├── pledge.txt
├── src/
│   ├── knn.py
│   ├── metrics.py
│   ├── splits.py
│   └── experiments.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_evaluation.ipynb
│   └── 03_cross_validation.ipynb
├── tests/
│   ├── test_knn.py
│   └── test_metrics.py
├── figures/
└── data/
    └── titanic3.csv
```

## Reproducibility
All random operations are seeded. The preprocessing pipeline fits imputation values on the training partition only; cross-validation fits preprocessing inside each training fold. The scaling experiment fits `StandardScaler` only on the training fold.
