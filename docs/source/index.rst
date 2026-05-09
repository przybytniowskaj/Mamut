MAMUT: Automated Tabular Classification
========================================

.. meta::
   :description: MAMUT is a Python package for transparent tabular classification with preprocessing, model selection, validation evidence, evaluation reports, and SHAP explanations.
   :keywords: MAMUT, automated machine learning, tabular classification, validation evidence, scikit-learn, XGBoost, Optuna, SHAP

MAMUT is a Python package for transparent classification workflows on tabular
data. It combines preprocessing, Optuna-driven model search, metric comparison,
validation evidence, configurable model artifacts, and HTML report generation
behind a compact API.

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
  stability, and confidence intervals.
* Configurable model artifacts for the best model and fitted candidate models.

Minimal Example
---------------

.. code-block:: python

   from sklearn.datasets import load_iris

   from mamut import Mamut

   X, y = load_iris(as_frame=True, return_X_y=True)

   mamut = Mamut(n_iterations=1, optimization_method="random_search")
   mamut.fit(X, y)

   predictions = mamut.predict(X)
   probabilities = mamut.predict_proba(X)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide
   reports
   benchmark_evidence
   notebooks/walkthrough

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   mamut
   mamut.preprocessing
   mamut.utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   dependency_policy
