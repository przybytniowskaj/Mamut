MAMUT: Automated Tabular Classification
========================================

.. meta::
   :description: MAMUT is a Python package for transparent tabular classification with preprocessing, model selection, validation evidence, evaluation reports, and SHAP explanations.
   :keywords: MAMUT, automated machine learning, tabular classification, validation evidence, scikit-learn, XGBoost, Optuna, SHAP

MAMUT is a Python package for transparent classification workflows on tabular
data. It combines preprocessing, Optuna-driven model search, metric comparison,
validation evidence, configurable model artifacts, and HTML report generation
behind a compact scikit-learn-style API.

Use MAMUT when you want a fast baseline for structured classification data and
a reproducible summary of the models, preprocessing decisions, metrics, plots,
validation diagnostics, and SHAP explanations produced during an experiment.

MAMUT is not an industrial AutoML replacement. It is most useful when readable
evidence, simple baselines, and validation integrity matter more than searching
the largest possible model space.

Highlights
----------

* Automated preprocessing for missing values, categorical variables, skewed
  numeric features, scaling, outliers, class imbalance, optional feature
  selection, and optional PCA.
* Model search across common scikit-learn classifiers and XGBoost.
* Hyperparameter optimization with Optuna using Bayesian or random search.
* Evaluation reports with metrics, confusion matrices, ROC curves, feature
  importances, and SHAP plots.
* Evidence diagnostics with leakage checks, baseline comparison, score
  stability, and approximate score intervals.
* Configurable model artifacts for the best model and fitted candidate models.

Start Here
----------

1. Install MAMUT from PyPI or create a repository development environment:
   :doc:`installation`.
2. Fit a first model on a small dataset and inspect predictions:
   :doc:`quickstart`.
3. Configure preprocessing, validation, final holdout data, and reproducibility:
   :doc:`user_guide`.
4. Generate and interpret HTML reports, artifacts, and evidence tables:
   :doc:`reports` and :doc:`benchmark_evidence`.

Minimal Example
---------------

.. code-block:: python

   from sklearn.datasets import load_iris

   from mamut import Mamut

   X, y = load_iris(as_frame=True, return_X_y=True)

   mamut = Mamut(
       n_iterations=1,
       optimization_method="random_search",
       holdout_size=0.2,
       random_state=42,
   )
   mamut.fit(X, y)

   predictions = mamut.predict(X.head())
   probabilities = mamut.predict_proba(X.head())
   report = mamut.evaluate(include_shap=False, write_html=False, save_plots=False)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Start Here

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide
   reports
   benchmark_evidence

.. toctree::
   :maxdepth: 2
   :caption: Examples

   notebooks/walkthrough

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   mamut
   mamut.preprocessing
   mamut.utils

.. toctree::
   :maxdepth: 1
   :caption: Project Maintenance

   dependency_policy
