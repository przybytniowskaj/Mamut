MAMUT: Automated Tabular Classification
========================================

.. meta::
   :description: MAMUT is a Python package for automated tabular classification with preprocessing, model selection, evaluation reports, and SHAP explanations.
   :keywords: MAMUT, automated machine learning, tabular classification, scikit-learn, XGBoost, Optuna, SHAP

MAMUT is a Python package for automated classification workflows on tabular
data. It combines preprocessing, Optuna-driven model search, metric comparison,
model persistence, and HTML report generation behind a compact API.

Use MAMUT when you want a fast baseline for structured classification data and
a reproducible summary of the models, preprocessing decisions, metrics, plots,
and SHAP explanations produced during an experiment.

Highlights
----------

* Automated preprocessing for missing values, categorical variables, skewed
  numeric features, scaling, outliers, class imbalance, optional feature
  selection, and optional PCA.
* Model search across common scikit-learn classifiers and XGBoost.
* Hyperparameter optimization with Optuna using Bayesian or random search.
* Evaluation reports with metrics, confusion matrices, ROC curves, feature
  importances, and SHAP plots.
* Saved model artifacts for the best model and fitted candidate models.

Minimal Example
---------------

.. code-block:: python

   from sklearn.datasets import load_iris

   from mamut.wrapper import Mamut

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
