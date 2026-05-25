Evidence Benchmark
==================

.. meta::
   :description: Reproduce MAMUT validation evidence diagnostics on public sklearn classification datasets.
   :keywords: MAMUT benchmark, validation evidence, baseline comparison, tabular classification

MAMUT includes a lightweight benchmark script for release diagnostics. The goal
is not to claim state-of-the-art AutoML performance. The goal is to verify that
the selected model beats trivial baselines, that stronger baselines are visible
when they challenge the selection, and that score stability is reported with
descriptive resampling intervals.

Run the benchmark from the repository root:

.. code-block:: sh

   uv run python scripts/benchmark_evidence.py --format markdown

The default run uses:

* sklearn ``breast_cancer``, ``digits``, and ``wine`` datasets
* fixed ``random_state=42``
* ``balanced_accuracy`` as the selection metric
* ``holdout_size=0.2`` for final evaluation
* final refit on all non-holdout modeling rows before holdout scoring
* one random-search iteration for speed
* three-fold repeated stratified CV with one repeat for score stability
* the lightweight ``quick`` candidate profile (logistic regression, random
  forest, extra trees, and Gaussian naive Bayes)

Example Diagnostic Output
-------------------------

The following output was generated from the locked development environment for
a release validation pass:

.. code-block:: text

   | dataset       | samples | features | classes | selected_model     | holdout_score | best_baseline       | best_baseline_score | repeated_cv_mean | repeated_cv_ci | guidance   | leakage_warnings |
   | ------------- | ------- | -------- | ------- | ------------------ | ------------- | ------------------- | ------------------- | ---------------- | -------------- | ---------- | ---------------- |
   | breast_cancer | 569     | 30       | 2       | RandomForestClassifier | 0.943         | Logistic Regression | 0.953               | 0.958            | [0.879, 1.000] | challenged | 0                |
   | digits        | 1797    | 64       | 10      | RandomForestClassifier | 0.969         | Random Forest       | 0.969               | 0.962            | [0.939, 0.986] | challenged | 0                |
   | wine          | 178     | 13       | 3       | LogisticRegression     | 1.000         | Logistic Regression | 1.000               | 0.981            | [0.934, 1.000] | confirmed  | 0                |

Interpretation
--------------

``confirmed`` means no evidence baseline exceeded the selected model by the
configured practical margin. ``challenged`` means a baseline matched or beat the
selected model strongly enough to require review. A challenge is useful signal:
it prevents MAMUT from presenting a validation-selected model as stronger than
the evidence supports.

On ``breast_cancer``, a simple logistic-regression baseline beats the selected
random-forest candidate on the holdout split, and MAMUT surfaces that
challenge. On ``digits``, a random-forest baseline matches the selected
candidate closely enough to require review. On ``wine``, the holdout score is
saturated while score-stability evidence remains visible. This is the intended
behavior: attractive holdout results do not suppress baseline or stability
checks.
