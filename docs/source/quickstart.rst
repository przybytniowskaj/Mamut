Quickstart
==========

.. meta::
   :description: Fit a first MAMUT classifier, generate predictions, inspect model comparison results, and create an HTML evaluation report.
   :keywords: MAMUT quickstart, automated classification, fit predict, model evaluation

This example trains MAMUT on the Iris dataset and keeps the search budget small
so it can run quickly on a local machine.

Fit a Model
-----------

.. code-block:: python

   from sklearn.datasets import load_iris

   from mamut.wrapper import Mamut

   X, y = load_iris(as_frame=True, return_X_y=True)

   mamut = Mamut(
       n_iterations=1,
       optimization_method="random_search",
       random_state=42,
   )
   mamut.fit(X, y)

MAMUT performs a stratified train/validation split, applies preprocessing,
compares candidate classifiers, tunes their hyperparameters, and stores the
validation-selected model in ``mamut.best_model_``.

Predict
-------

.. code-block:: python

   predictions = mamut.predict(X.head())
   probabilities = mamut.predict_proba(X.head())

``predict`` returns predicted classes. ``predict_proba`` returns class
probabilities from the selected best model.

Inspect Results
---------------

.. code-block:: python

   mamut.best_score_
   mamut.validation_summary_
   mamut.optuna_studies_.keys()

``validation_summary_`` contains per-model validation metric scores and training
durations. ``training_summary_`` remains available as a backward-compatible
alias.
``optuna_studies_`` stores the optimization study for each fitted model.

Use a Final Holdout
-------------------

For an unbiased final report score, reserve holdout data that is not used for
model or ensemble selection:

.. code-block:: python

   mamut = Mamut(
       n_iterations=1,
       optimization_method="random_search",
       holdout_size=0.2,
       random_state=42,
   )
   mamut.fit(X, y)
   mamut.holdout_summary_

Generate a Report
-----------------

.. code-block:: python

   mamut.evaluate(n_top_models=3)

The report is written to ``mamut_report/`` in the current working directory.
The method uses the holdout split automatically when one is available;
otherwise, it clearly reports validation metrics. Generated plots are stored
under ``mamut_report/plots/``.

Save the Best Model
-------------------

Create the output directory first, then save the selected best model:

.. code-block:: python

   from pathlib import Path

   output_dir = Path("saved_models")
   output_dir.mkdir(exist_ok=True)

   mamut.save_best_model(str(output_dir))
