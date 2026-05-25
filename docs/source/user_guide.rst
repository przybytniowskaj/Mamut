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
categorical values, one-hot encoding, native categorical columns for supported
boosting models, skew correction, scaling, optional outlier filtering,
imbalanced target resampling, optional feature selection, and optional PCA.

By default, ``preprocessing_profile="auto"`` lets each candidate use a
model-aware preprocessing profile. Linear, kernel, and distance-based models
use the generic one-hot path. Tree models use one-hot encoded categoricals
without unnecessary numeric scaling. CatBoost and LightGBM use native
categorical columns when compatible with the rest of the preprocessing options.
Set ``preprocessing_profile="generic_ohe"`` to force the legacy shared one-hot
path for every candidate.

Automatic row removal for outliers is disabled by default because it can change
the target distribution and hurt external validity. Enable it only when that is
part of the intended experiment:

.. code-block:: python

   mamut = Mamut(outlier_removal=True)

Model Search
------------

MAMUT compares a set of supported classifiers and selects the best model by the
configured score metric on a validation split. Supported model families include
logistic regression, random forests, extremely randomized trees, histogram
gradient boosting, XGBoost, LightGBM, CatBoost, support vector machines,
multilayer perceptrons, Gaussian naive Bayes, and k-nearest neighbors.

Use ``search_profile`` to choose the candidate pool:

.. code-block:: python

   mamut = Mamut(search_profile="quick")      # small fast pool
   mamut = Mamut(search_profile="balanced")   # default tabular pool
   mamut = Mamut(search_profile="thorough")   # includes slower learners

Use ``include_models`` for an exact candidate set, or ``exclude_models`` to
remove expensive or unwanted estimators by class name:

.. code-block:: python

   mamut = Mamut(include_models=["RandomForestClassifier", "LGBMClassifier"])
   mamut = Mamut(exclude_models=["SVC", "MLPClassifier"])

``include_models`` and ``exclude_models`` are mutually exclusive. Set
``n_jobs`` to control parallelism for supported estimators.

Selection Strategy
------------------

The default ``selection_strategy="single_split"`` keeps runtime low by choosing
the best tuned candidate on one validation split. For higher-integrity model
development, use nested validation over the non-holdout modeling data:

.. code-block:: python

   mamut = Mamut(
       search_profile="balanced",
       selection_strategy="nested_cv",
       selection_cv_splits=5,
       selection_cv_repeats=2,
       selection_practical_margin=0.005,
   )

Nested-CV selection performs tuning inside every outer training fold, fits
preprocessing inside those folds, and never uses final holdout rows. It selects
by mean outer-fold score; models within the practical margin are treated as
ties and resolved by lower score variance, then faster selection runtime.
``selection_strategy="repeated_cv"`` is retained only as a deprecated alias.
Inspect ``selection_summary_`` after ``fit`` to see the selection evidence.

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
Set ``verbose=True`` when you want model-search progress logging and Optuna
progress bars.

Metrics
-------

Choose a score metric with ``score_metric``:

.. code-block:: python

   mamut = Mamut(score_metric="balanced_accuracy")

Supported values are ``accuracy``, ``precision``, ``recall``, ``f1``,
``balanced_accuracy``, ``jaccard``, and ``roc_auc_score``. Classification
metrics are weighted when needed for multiclass problems.

Validation and Holdout Data
---------------------------

By default, ``fit`` creates a stratified train/validation split. The validation
split is used for model selection, ensemble selection, and
``validation_summary_``:

.. code-block:: python

   mamut = Mamut(validation_size=0.2, random_state=42)
   mamut.fit(X, y)

For final evaluation, reserve a holdout set that is never used during model or
ensemble selection:

.. code-block:: python

   mamut = Mamut(holdout_size=0.2, random_state=42)
   mamut.fit(X, y)
   mamut.evaluate()  # uses the holdout split automatically

You can also provide an explicit holdout set:

.. code-block:: python

   mamut.fit(X_train, y_train, X_holdout=X_holdout, y_holdout=y_holdout)

Use holdout scores for final reporting. Use validation scores for model
selection and debugging.

When observations share a subject, household, session, patient, or other unit,
pass group identifiers so no related rows cross validation boundaries:

.. code-block:: python

   mamut = Mamut(
       selection_strategy="nested_cv",
       holdout_size=0.2,
       refit_final_model=True,
   )
   mamut.fit(X, y, groups=passenger_group)

For an explicit holdout, also pass ``groups_holdout=``; overlapping modeling
and holdout groups are rejected.

Final Refit
-----------

By default, ``best_model_`` is the estimator selected on the validation split.
This keeps the selected model aligned with the validation evidence. If you want
the public prediction pipeline to refit on all non-holdout modeling data after
selection, set ``refit_final_model=True``:

.. code-block:: python

   mamut = Mamut(
       holdout_size=0.2,
       refit_final_model=True,
       random_state=42,
   )
   mamut.fit(X, y)

The final refit never uses holdout rows. Use this option for deployment
artifacts after you have accepted validation diagnostics. With
``selection_strategy="nested_cv"``, the selected model family is also retuned
by cross-validation on all non-holdout modeling rows before the final fit.

Prediction Contract
-------------------

``Mamut.predict`` and ``mamut.best_model_.predict`` return the original target
labels, even though MAMUT encodes labels internally for estimator training.
``predict_proba`` returns estimator probabilities in the class order exposed by
the fitted public model. Unknown categorical levels at prediction time are
encoded as all zeros for that categorical feature group instead of raising an
error.

Evidence Checks
---------------

``evaluate`` includes an evidence layer by default. It is designed to answer
whether the reported model score is trustworthy enough to take seriously, not
only which model has the largest score.

The evidence layer includes:

* basic leakage checks for target-like columns, exact target copies, identifier
  columns, duplicate feature rows, and class imbalance
* comparison against fitted MAMUT candidates plus dummy, logistic regression,
  and random forest baselines
* repeated stratified cross-validation, or group-disjoint stratified folds when
  ``groups=`` is supplied, for score stability
* descriptive t-based stability intervals over repeated fold scores, clipped
  to the valid metric range; these folds are dependent and the interval is not
  a confirmatory confidence claim
* evidence-guided selection guidance that confirms, challenges, or blocks trust
  in the validation-selected model

.. code-block:: python

   mamut = Mamut(
       holdout_size=0.2,
       evidence_cv_splits=5,
       evidence_cv_repeats=3,
       evidence_confidence_level=0.95,
   )
   mamut.fit(X, y)
   mamut.evaluate()

You can compute the evidence tables without writing a report:

.. code-block:: python

   evidence = mamut.generate_evidence()
   mamut.baseline_comparison_
   mamut.score_stability_
   mamut.leakage_checks_
   mamut.selection_guidance_

For a locked final holdout confirmation, avoid comparing alternate MAMUT
candidates on that holdout:

.. code-block:: python

   evidence = mamut.generate_evidence(
       dataset="holdout",
       include_candidate_comparison=False,
   )

For lightweight evaluation in scripts or CI, disable expensive or file-writing
outputs while keeping evidence generation enabled:

.. code-block:: python

   result = mamut.evaluate(
       include_shap=False,
       write_html=False,
       save_plots=False,
   )

The score stability check refits the selected estimator and baseline models
with fold-local preprocessing. It does not retune hyperparameters inside each
fold, so treat it as a stability diagnostic rather than a full nested
cross-validation benchmark. Use ``selection_strategy="nested_cv"`` when model
selection itself needs nested evaluation.

The evidence-guided selection table is intentionally conservative. If a
baseline beats the selected model on final holdout data, MAMUT challenges the
selection for review but keeps the selected candidate as the recommendation.
It does not silently promote the holdout winner. Use that challenge
to rerun model selection or reserve a new final holdout before deployment.

For a reproducible example of these diagnostics on public sklearn datasets, see
:doc:`benchmark_evidence`.

Reproducibility
---------------

Pass ``random_state`` to control the train/validation/holdout split,
preprocessing components, resampling, and supported model initializers:

.. code-block:: python

   mamut = Mamut(random_state=42)

Fitted candidate models are kept in memory by default. Set
``save_models=True`` to write them under ``fitted_models/<timestamp>/``:

.. code-block:: python

   mamut = Mamut(save_models=True)

Because this directory is created relative to the current working directory,
run experiments from a known project or experiment folder.

Limitations
-----------

MAMUT currently targets supervised classification only. It is designed for
tabular data and does not implement time-series validation, regression,
multilabel classification, text pipelines, image pipelines, or custom model
registries. It should be treated as a transparent baseline and reporting
assistant, not as a replacement for larger AutoML systems.
