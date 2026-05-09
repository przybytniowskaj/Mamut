User Guide
==========

.. meta::
   :description: Learn how MAMUT handles tabular classification data, preprocessing, model search, metrics, and reproducibility.
   :keywords: MAMUT user guide, tabular classification, preprocessing, model selection, Optuna

MAMUT exposes the main workflow through :class:`mamut.wrapper.Mamut`. The class
expects tabular features in a pandas ``DataFrame`` and a categorical target in a
pandas ``Series`` or compatible array.

Data Requirements
-----------------

* ``X`` should be a pandas ``DataFrame`` with numeric and/or categorical
  feature columns.
* ``y`` must represent classes. Floating point targets are rejected because
  MAMUT is a classification package, not a regression package.
* With preprocessing enabled, MAMUT detects numeric and categorical columns
  automatically unless ``numeric_features`` or ``categorical_features`` are
  passed explicitly.

Preprocessing
-------------

Preprocessing is enabled by default with ``preprocess=True``. Extra keyword
arguments passed to ``Mamut`` are forwarded to
:class:`mamut.preprocessing.preprocessing.Preprocessor`.

.. code-block:: python

   mamut = Mamut(
       num_imputation="mean",
       cat_imputation="most_frequent",
       scaling="standard",
       feature_selection=True,
       pca=False,
   )

The preprocessing pipeline can handle missing numeric values, missing
categorical values, one-hot encoding, skew correction, scaling, outlier
filtering, imbalanced target resampling, optional feature selection, and
optional PCA.

Model Search
------------

MAMUT compares a set of supported classifiers and selects the best model by the
configured score metric. Supported model families include logistic regression,
random forest, support vector machines, XGBoost, multilayer perceptrons,
Gaussian naive Bayes, and k-nearest neighbors.

Use ``exclude_models`` to remove expensive or unwanted estimators by class name:

.. code-block:: python

   mamut = Mamut(exclude_models=["SVC", "MLPClassifier"])

Hyperparameter Search
---------------------

Set the optimization method and iteration budget at initialization:

.. code-block:: python

   mamut = Mamut(
       optimization_method="bayes",
       n_iterations=30,
       random_state=42,
   )

Use ``optimization_method="random_search"`` for a simpler random search. Use
``optimization_method="bayes"`` for Optuna's tree-structured Parzen estimator.

Metrics
-------

Choose a score metric with ``score_metric``:

.. code-block:: python

   mamut = Mamut(score_metric="balanced_accuracy")

Supported values are ``accuracy``, ``precision``, ``recall``, ``f1``,
``balanced_accuracy``, ``jaccard``, and ``roc_auc_score``. Classification
metrics are weighted when needed for multiclass problems.

Reproducibility
---------------

Pass ``random_state`` to control the train/test split, preprocessing components,
resampling, and supported model initializers:

.. code-block:: python

   mamut = Mamut(random_state=42)

The fitted candidate models are saved under ``fitted_models/<timestamp>/``.
Because this directory is created relative to the current working directory, run
experiments from a known project or experiment folder.

Limitations
-----------

MAMUT currently targets supervised classification only. It is designed for
tabular data and does not implement time-series validation, regression,
multilabel classification, text pipelines, image pipelines, or custom model
registries.
