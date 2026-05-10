Evidence Benchmark
==================

.. meta::
   :description: Reproduce MAMUT validation evidence diagnostics on public sklearn classification datasets.
   :keywords: MAMUT benchmark, validation evidence, baseline comparison, tabular classification

MAMUT includes a lightweight benchmark script for release diagnostics. The goal
is not to claim state-of-the-art AutoML performance. The goal is to verify that
the selected model beats trivial baselines, that stronger baselines are visible
when they challenge the selection, and that score stability is reported with
confidence intervals.

Run the benchmark from the repository root:

.. code-block:: sh

   uv run python scripts/benchmark_evidence.py --format markdown

The default run uses:

* sklearn ``breast_cancer``, ``digits``, and ``wine`` datasets
* fixed ``random_state=42``
* ``balanced_accuracy`` as the selection metric
* ``holdout_size=0.2`` for final evaluation
* one random-search iteration for speed
* three-fold repeated stratified CV with one repeat for score stability
* a lightweight candidate set that excludes SVC, MLP, and XGBoost

Current Diagnostic Output
-------------------------

The following output was generated from the locked development environment for
the ``0.2.0`` release pass:

.. code-block:: text

   | dataset       | samples | features | classes | selected_model     | holdout_score | best_baseline       | best_baseline_score | repeated_cv_mean | repeated_cv_ci | guidance   | leakage_warnings |
   | ------------- | ------- | -------- | ------- | ------------------ | ------------- | ------------------- | ------------------- | ---------------- | -------------- | ---------- | ---------------- |
   | breast_cancer | 569     | 30       | 2       | LogisticRegression | 0.960         | Logistic Regression | 0.948               | 0.968            | [0.928, 1.000] | confirmed  | 0                |
   | digits        | 1797    | 64       | 10      | LogisticRegression | 0.949         | Random Forest       | 0.972               | 0.948            | [0.932, 0.965] | challenged | 0                |
   | wine          | 178     | 13       | 3       | LogisticRegression | 1.000         | Logistic Regression | 1.000               | 0.964            | [0.886, 1.000] | challenged | 0                |

Interpretation
--------------

``confirmed`` means no evidence baseline exceeded the selected model by the
configured practical margin. ``challenged`` means a baseline matched or beat the
selected model strongly enough to require review. A challenge is useful signal:
it prevents MAMUT from presenting a validation-selected model as stronger than
the evidence supports.

On ``digits``, Random Forest performs better on holdout than the selected
Logistic Regression candidate. MAMUT reports this instead of silently changing
the chosen model after looking at holdout data. On ``wine``, the holdout score
is saturated, but repeated validation still challenges the selected candidate.
That is the intended behavior: small datasets with perfect holdout scores still
need stability checks.
