MAMUT API
=========

.. meta::
   :description: API reference for the MAMUT automated tabular classification workflow.
   :keywords: MAMUT API, Mamut class, automated classification, Python API

The main public API is ``Mamut``. Import it with ``from mamut import Mamut`` to
fit candidate models, select the best pipeline, generate predictions, create
reports, and save model artifacts.

The public prediction surface is ``best_model_``, ``predict``, and
``predict_proba``. Internal fitted estimators are also exposed for inspection
through attributes such as ``raw_fitted_models_`` and ``validation_summary_``,
but these are primarily diagnostics for reports and model comparison.

Mamut
-----

.. autoclass:: mamut.wrapper.Mamut
   :show-inheritance:

Common Methods
~~~~~~~~~~~~~~

.. automethod:: mamut.wrapper.Mamut.fit
   :no-index:

.. automethod:: mamut.wrapper.Mamut.predict
   :no-index:

.. automethod:: mamut.wrapper.Mamut.predict_proba
   :no-index:

.. automethod:: mamut.wrapper.Mamut.evaluate
   :no-index:

.. automethod:: mamut.wrapper.Mamut.generate_evidence
   :no-index:

.. automethod:: mamut.wrapper.Mamut.save_best_model
   :no-index:

.. automethod:: mamut.wrapper.Mamut.create_ensemble
   :no-index:

.. automethod:: mamut.wrapper.Mamut.create_greedy_ensemble
   :no-index:

Model Selection
---------------

``ModelSelector`` is used internally by :class:`mamut.wrapper.Mamut` to compare
supported estimators and optimize hyperparameters. Most users should configure
model search through ``Mamut`` instead of instantiating ``ModelSelector``
directly.

.. autoclass:: mamut.model_selection.ModelSelector
   :members:
   :show-inheritance:

Evaluation
----------

``ModelEvaluator`` is used internally by ``Mamut.evaluate`` to produce the HTML
report and plots.

.. autoclass:: mamut.evaluation.ModelEvaluator
   :members:
   :show-inheritance:

Evidence
--------

Evidence helpers power the validation integrity, leakage, baseline, and score
stability sections of ``Mamut.evaluate``.

.. automodule:: mamut.evidence
   :members:
