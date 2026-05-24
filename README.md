<div align="center">
  <img src="https://raw.githubusercontent.com/przybytniowskaj/Mamut/main/docs/source/_static/logo.webp" alt="MAMUT Logo" width="180" />
  <h1>MAMUT</h1>
  <p>Machine Automated Modelling and Utility Toolkit for tabular classification.</p>
</div>

[![Documentation Status](https://readthedocs.org/projects/mamut/badge/?version=latest)](https://mamut.readthedocs.io/en/latest/?badge=latest)
[![Test Pipeline](https://github.com/przybytniowskaj/Mamut/actions/workflows/tests.yml/badge.svg)](https://github.com/przybytniowskaj/Mamut/actions/workflows/tests.yml)
[![Pre-commit Pipeline](https://github.com/przybytniowskaj/Mamut/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/przybytniowskaj/Mamut/actions/workflows/pre-commit.yaml)
[![Security Audit](https://github.com/przybytniowskaj/Mamut/actions/workflows/security.yml/badge.svg)](https://github.com/przybytniowskaj/Mamut/actions/workflows/security.yml)
![License](https://img.shields.io/github/license/przybytniowskaj/Mamut)

## Overview
MAMUT is a Python toolkit for transparent **classification** workflows on tabular data. It bundles preprocessing, Optuna-driven hyperparameter optimization, model comparison, validation diagnostics, and reporting into a single workflow built on scikit-learn, XGBoost, LightGBM, and CatBoost.

MAMUT is best used as a readable baseline and experiment report generator for beginners, small teams, and portfolio-scale projects. It is not positioned as a replacement for industrial AutoML systems such as AutoGluon, FLAML, or H2O AutoML; its value is in showing what was tried, how the result was validated, and whether simple baselines challenge the selected model.

## Key Features
- End-to-end preprocessing: missing values, categorical encoding, skew correction, scaling, optional outlier filtering, imbalance handling (SMOTE/undersampling/SMOTETomek), optional feature selection, and PCA.
- Model search across common and boosted classifiers, including LogisticRegression, RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier, GaussianNB, SVC, MLPClassifier, and KNeighborsClassifier.
- Hyperparameter optimization with Optuna (TPE/Bayesian or random search).
- Validation-based model selection with optional nested CV, group-disjoint splits, and final holdout evaluation.
- Evidence reporting: leakage checks, fitted-candidate comparison, dummy/logistic/random-forest baselines, and descriptive score-stability intervals from repeated stratified or grouped CV.
- Report generation via `evaluate()` with metrics, plots, and SHAP explanations.
- Configurable artifacts: `fit()` keeps models in memory by default and saves fitted models only when `save_models=True`.
- Reproducible benchmark diagnostics via `scripts/benchmark_evidence.py`.
- External Kaggle benchmark harness with development/confirmation separation and explicit submission controls.

## Installation
Python 3.12 is the target runtime (see `.python-version`).

From PyPI:
```sh
pip install mamut
```

From source:
```sh
git clone https://github.com/przybytniowskaj/Mamut.git
cd Mamut
uv sync --all-groups
```

For development with uv:
```sh
uv sync --all-groups
```

## Quickstart
```python
from sklearn.datasets import load_iris
from mamut import Mamut

X, y = load_iris(as_frame=True, return_X_y=True)

mamut = Mamut(
    n_iterations=1,
    optimization_method="random_search",
    holdout_size=0.2,
    refit_final_model=True,
    random_state=42,
)
mamut.fit(X, y)

preds = mamut.predict(X.head())
proba = mamut.predict_proba(X.head())
report = mamut.evaluate(include_shap=False, write_html=False, save_plots=False)
```

## Configuration Notes
- With preprocessing enabled (default), pass `X` as a pandas `DataFrame` and `y` as a `Series`.
- Targets must be categorical (float targets raise a `ValueError`).
- `fit()` performs a stratified train/validation split controlled by `validation_size` and `random_state`; pass `groups=` to keep related rows in the same partition.
- Set `holdout_size` or pass `X_holdout`/`y_holdout` to reserve final evaluation data that is not used for model or ensemble selection.
- Select the optimization strategy with `optimization_method="bayes"` or `"random_search"`.
- Control the search budget with `n_iterations`.
- Choose a model pool with `search_profile="quick"`, `"balanced"`, or `"thorough"`.
- Include exact models with `include_models=["RandomForestClassifier", "LGBMClassifier"]`, or exclude models by class name with `exclude_models=["SVC"]`.
- Use `selection_strategy="nested_cv"` for fold-local tuning and model comparison when runtime matters less than selection confidence.
- Control parallelism for supported estimators with `n_jobs`.
- `preprocessing_profile="auto"` lets tree and native-boosting candidates use model-aware preprocessing; set `preprocessing_profile="generic_ohe"` to force the legacy shared one-hot path.
- Preprocessing options are passed directly into `Mamut(...)` (e.g., `pca=True`, `feature_selection=True`, `num_imputation="knn"`). Native categorical preprocessing falls back to one-hot when PCA, feature selection, or imbalance resampling requires numeric arrays.
- Automatic outlier row removal is disabled by default; enable it explicitly with `outlier_removal=True`.
- Unknown categorical levels at prediction time are ignored by one-hot profiles instead of failing the prediction.
- Set `verbose=True` to show model-search progress logging and Optuna progress bars.
- Use `save_models=True` to write fitted candidate pipelines under `./fitted_models/<timestamp>/`.
- Use `refit_final_model=True` only after you accept validation and holdout diagnostics; the final refit uses all non-holdout modeling data and never uses holdout rows.
- `score_metric` expects one of: `accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `jaccard`, `roc_auc_score`.
- Configure descriptive evidence stability checks with `evidence_cv_splits`, `evidence_cv_repeats`, and `evidence_confidence_level`; dependent CV fold intervals are not final confidence claims.

## Outputs and Reports
- `mamut.best_model_`: validation-selected public prediction pipeline after `fit`; `predict()` returns original target labels.
- `mamut.validation_summary_`: per-model validation scores and timings.
- `mamut.selection_summary_`: model-selection diagnostics for the active selection strategy.
- `mamut.fitted_preprocessors_`: fitted candidate-specific preprocessors keyed by model name.
- `mamut.holdout_summary_`: optional final holdout diagnostic scores when holdout data is configured; use a refitted selected model for final deployment scoring.
- `mamut.evidence_report_`: validation integrity, evidence-guided selection, leakage checks, baseline comparison, and score stability tables generated by `evaluate()` or `generate_evidence()`.
- Use `generate_evidence(dataset="holdout", include_candidate_comparison=False)` for a locked final confirmation report that does not score alternate MAMUT candidates on the holdout.
- `mamut.optuna_studies_`: Optuna studies keyed by model name.
- `mamut.evaluate()`: writes an HTML report to `./mamut_report/report_<timestamp>.html` and stores plots in `./mamut_report/plots/`. It uses holdout data automatically when available and includes evidence sections by default.
- `mamut.evaluate(include_shap=False, write_html=False, save_plots=False)`: run lightweight evaluation without expensive SHAP or file artifacts.
- `mamut.save_best_model(path)`: writes the best model to an existing directory as a `.joblib` file.

## Development
```sh
uv sync --all-groups
uv run deptry .
scripts/audit_dependencies.sh
uv run pytest
uv run pre-commit run --all-files
uv run make -C docs html
uv run sphinx-build -W --keep-going -b html docs/source docs/build/html-strict
uv run python scripts/benchmark_evidence.py
uv run python scripts/benchmark_kaggle.py spaceship-titanic --stage development --runs 1 --n-iterations 1
uv build
uv run twine check dist/*
```

## Documentation
- Documentation site: https://mamut.readthedocs.io/en/latest/
- Quickstart: https://mamut.readthedocs.io/en/latest/quickstart.html
- User guide: https://mamut.readthedocs.io/en/latest/user_guide.html
- Reports and artifacts: https://mamut.readthedocs.io/en/latest/reports.html
- Evidence benchmark: https://mamut.readthedocs.io/en/latest/benchmark_evidence.html
- Kaggle benchmarks: https://mamut.readthedocs.io/en/latest/kaggle_benchmarks.html
- API reference: https://mamut.readthedocs.io/en/latest/mamut.html
- Notebook walkthrough: `docs/source/notebooks/walkthrough.ipynb`
- Changelog: `CHANGELOG.md`

## License
MIT. See `LICENSE`.
