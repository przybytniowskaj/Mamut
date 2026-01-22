<div align="center">
  <img src="docs/source/_static/logo.webp" alt="MAMUT Logo" width="180" />
  <h1>MAMUT</h1>
  <p>Machine Automated Modelling and Utility Toolkit for tabular classification.</p>
</div>

[![Documentation Status](https://readthedocs.org/projects/mamut/badge/?version=latest)](https://mamut.readthedocs.io/en/latest/?badge=latest)
[![Test Pipeline](https://github.com/przybytniowskaj/Mamut/actions/workflows/tests.yml/badge.svg)](https://github.com/przybytniowskaj/Mamut/actions/workflows/tests.yml)
[![Pre-commit Pipeline](https://github.com/przybytniowskaj/Mamut/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/przybytniowskaj/Mamut/actions/workflows/pre-commit.yaml)
![License](https://img.shields.io/github/license/przybytniowskaj/Mamut)

## Overview
MAMUT is a Python toolkit that automates model selection and evaluation for **classification** tasks on tabular data. It bundles preprocessing, Optuna-driven hyperparameter optimization, model comparison, and reporting into a single workflow built on scikit-learn and XGBoost.

## Key Features
- End-to-end preprocessing: missing values, categorical encoding, skew correction, scaling, outlier filtering, imbalance handling (SMOTE/undersampling/SMOTETomek), optional feature selection, and PCA.
- Model search across common classifiers (LogisticRegression, RandomForestClassifier, SVC, XGBClassifier, MLPClassifier, GaussianNB, KNeighborsClassifier).
- Hyperparameter optimization with Optuna (TPE/Bayesian or random search).
- Report generation via `evaluate()` with metrics, plots, and SHAP explanations.
- Saved artifacts: `fit()` stores fitted models; `evaluate()` writes an HTML report and plots to disk.

## Installation
Python 3.12 is the target runtime (see `.python-version`).

From PyPI:
```sh
pip install mamut
```

From source:
```sh
pip install -e .
```

For development with Poetry:
```sh
poetry install
```

## Quickstart
```python
from sklearn.datasets import load_iris
from mamut.wrapper import Mamut

X, y = load_iris(as_frame=True, return_X_y=True)

mamut = Mamut(n_iterations=5, optimization_method="bayes")
mamut.fit(X, y)

preds = mamut.predict(X)
proba = mamut.predict_proba(X)
```

## Configuration Notes
- With preprocessing enabled (default), pass `X` as a pandas `DataFrame` and `y` as a `Series`.
- Targets must be categorical (float targets raise a `ValueError`).
- `fit()` performs a stratified 80/20 train/test split controlled by `random_state`.
- Select the optimization strategy with `optimization_method="bayes"` or `"random_search"`.
- Control the search budget with `n_iterations`.
- Exclude models by class name (e.g., `exclude_models=["SVC"]`).
- Preprocessing options are passed directly into `Mamut(...)` (e.g., `pca=True`, `feature_selection=True`, `num_imputation="knn"`).
- `score_metric` expects one of: `accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `jaccard`, `roc_auc_score`.

## Outputs and Reports
- `mamut.best_model_`: best performing pipeline after `fit`.
- `mamut.training_summary_`: per-model scores and timings.
- `mamut.optuna_studies_`: Optuna studies keyed by model name.
- `mamut.evaluate()`: writes an HTML report to `./mamut_report/report_<timestamp>.html` and stores plots in `./mamut_report/plots/`.
- `mamut.save_best_model(path)`: writes the best model to an existing directory as a `.joblib` file.
- `fit()` saves all fitted models to `./fitted_models/<timestamp>/` as `.joblib` files.

## Development
```sh
poetry run pytest
poetry run pre-commit run --all-files
make -C docs html
```

## Examples and Docs
- Notebooks: `walkthrough.ipynb` and `docs/source/notebooks/walkthrough.ipynb`.
- Documentation site: https://mamut.readthedocs.io/en/latest/

## License
MIT. See `LICENSE`.
