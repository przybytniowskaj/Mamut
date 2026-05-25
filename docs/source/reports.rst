Reports and Artifacts
=====================

.. meta::
   :description: Understand MAMUT evaluation reports, generated plots, SHAP explanations, saved fitted models, and output directories.
   :keywords: MAMUT reports, SHAP, model artifacts, evaluation report, fitted models

MAMUT can write experiment artifacts to disk during fitting and evaluation.
These outputs are useful for inspecting model behavior, comparing candidates,
and sharing experiment results.

Fitted Models
-------------

Calling ``fit`` keeps fitted candidate pipelines in memory by default. To save
them to disk, initialize MAMUT with ``save_models=True``:

.. code-block:: python

   mamut = Mamut(save_models=True)
   mamut.fit(X, y)

The fitted candidate pipelines are then saved under:

.. code-block:: text

   fitted_models/<timestamp>/

Each model is serialized as a ``.joblib`` file. The selected best pipeline is
also available in memory as ``mamut.best_model_``.

To save only the selected best model, create an output directory and call
``save_best_model``:

.. code-block:: python

   from pathlib import Path

   output_dir = Path("saved_models")
   output_dir.mkdir(exist_ok=True)

   mamut.save_best_model(str(output_dir))

Evaluation Report
-----------------

Call ``evaluate`` after ``fit``. When a holdout set is configured, ``evaluate``
uses it automatically. Otherwise, the report clearly shows validation metrics:

.. code-block:: python

   result = mamut.evaluate(n_top_models=3)
   result["report_path"]

Use a custom output directory when running multiple experiments:

.. code-block:: python

   result = mamut.evaluate(output_dir="reports/run_001")
   result["report_path"]

Evidence sections are included by default. Disable them only when report speed
matters more than validation diagnostics:

.. code-block:: python

   mamut.evaluate(include_evidence=False)

SHAP and file artifacts can also be disabled for lightweight validation runs:

.. code-block:: python

   result = mamut.evaluate(
       include_shap=False,
       write_html=False,
       save_plots=False,
   )
   result["evaluation_dataset"]

By default, the HTML report is written to:

.. code-block:: text

   mamut_report/report_<timestamp>.html

Generated plots are written to:

.. code-block:: text

   mamut_report/plots/

Report Contents
---------------

The report includes:

* system and Python environment details
* dataset size, feature overview, missing rows, and class distribution
* validation or holdout model comparison metrics and training durations
* validation integrity checks
* evidence-guided selection guidance
* leakage risk checks
* dummy, logistic regression, and random forest baseline comparison
* repeated stratified cross-validation score stability, using group-disjoint
  folds when groups were supplied, with descriptive resampling intervals that
  are not confirmatory confidence intervals
* ROC curve plots when ``save_plots=True``
* confusion matrices when ``save_plots=True``
* Optuna optimization history plots when studies are available
* feature importance plots when supported by the selected models
* SHAP beeswarm plots when ``include_shap=True`` and ``save_plots=True``
* preprocessing steps recorded by the fitted preprocessor

SHAP Explanations
-----------------

MAMUT uses SHAP during report generation when ``include_shap=True`` and plots
are being saved. Tree-like models generally use the model directly. Some
estimators are explained through their prediction function. SHAP computation can
be slower than basic prediction, so report generation may take longer than
fitting for some model/data combinations. Use ``shap_max_samples`` to cap the
number of rows used for explanations.

Output Directory Notes
----------------------

All output paths are relative to the current working directory. For clean
experiments, run MAMUT from a dedicated directory or move artifacts after each
run. Generated folders such as ``fitted_models/`` and ``mamut_report/`` should
not be committed to the repository.
