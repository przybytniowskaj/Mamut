Quickstart
==========

.. meta::
   :description: Fit a first MAMUT classifier, generate predictions, inspect model comparison results, and create an HTML evaluation report.
   :keywords: MAMUT quickstart, automated classification, fit predict, model evaluation

This example trains MAMUT on the Iris dataset, reserves a final holdout split,
and keeps the search budget small so it can run quickly on a local machine.

Fit a Model
-----------

.. code-block:: python

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

MAMUT performs stratified train/validation splitting inside the modeling data,
applies preprocessing, compares candidate classifiers, tunes their
hyperparameters, and stores the selected public prediction pipeline in
``mamut.best_model_``. With ``refit_final_model=True``, that pipeline is refit
on all non-holdout modeling rows. The holdout split is kept out of selection
and is used only for final evaluation.

Predict
-------

.. code-block:: python

   predictions = mamut.predict(X.head())
   probabilities = mamut.predict_proba(X.head())

``predict`` returns predicted classes in the original target labels.
``predict_proba`` returns class probabilities from the selected best model.

Inspect Results
---------------

.. code-block:: python

   mamut.best_score_
   mamut.validation_summary_
   mamut.holdout_summary_
   mamut.optuna_studies_.keys()

``validation_summary_`` contains per-model validation metric scores and training
durations. ``training_summary_`` remains available as a backward-compatible
alias.
``optuna_studies_`` stores the optimization study for each fitted model.
``holdout_summary_`` contains holdout diagnostics when holdout data is
configured; the selected-model row is the final refit score in this example.

Generate a Report
-----------------

.. code-block:: python

   mamut.evaluate(n_top_models=3)

By default, the report is written to ``mamut_report/`` in the current working
directory and plots are stored under ``mamut_report/plots/``. The method uses
the holdout split automatically when one is available; otherwise, it clearly
reports validation metrics. Evidence sections include validation integrity,
leakage checks, baseline comparison, and repeated validation score stability.

For a fast smoke run without SHAP or files, keep the evidence tables in memory:

.. code-block:: python

   result = mamut.evaluate(
       n_top_models=3,
       include_shap=False,
       write_html=False,
       save_plots=False,
   )
   result["evaluation_dataset"]

Save the Best Model
-------------------

Create the output directory first, then save the selected best model:

.. code-block:: python

   from pathlib import Path

   output_dir = Path("saved_models")
   output_dir.mkdir(exist_ok=True)

   mamut.save_best_model(str(output_dir))
