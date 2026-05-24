Kaggle Benchmarks
=================

.. meta::
   :description: Run reproducible MAMUT Spaceship Titanic benchmark studies with locked confirmation data and explicit Kaggle submission control.
   :keywords: MAMUT Kaggle benchmark, Spaceship Titanic, grouped validation, tabular classification

MAMUT includes a competition-specific benchmark harness for
``spaceship-titanic``. It is designed to test the package against a realistic
external dataset without confusing local validation with official Kaggle
leaderboard results. No MAMUT leaderboard result should be claimed until an
uploaded submission reference and public score are recorded.

Protocol
--------

The default ``development`` stage reserves a deterministic, group-disjoint
confirmation partition and excludes it from experimentation. Development runs
use passenger-group-disjoint outer evaluation and nested, fold-local model
selection. Only a frozen candidate may be run once with ``--stage
confirmation``. A confirmation score is an observation of that candidate; it
must not select a different model. Confirmation fits on development data
without exposing reserved labels to candidate-reporting logic, then predicts
the reserved partition once for the selected candidate.

The default ``spaceship_inductive_v2`` recipe derives row-observable domain
features including cabin structure, spending behavior, age groups, CryoSleep
consistency, and family name. ``spaceship_cohort_v2`` additionally derives
batch-level family and passenger-group sizes. It requires
``--group-scope household_component`` so related surnames cannot cross
validation folds. Older recipes remain available only for reproducibility.

Reported development metrics include outer accuracy, fixed-baseline uplift,
audit-only alternate-candidate deltas, and a group-bootstrap interval over
recorded outer predictions. Alternate outer-holdout scores can inform the next
development run, but are not retrospective model selection. The interval
reflects evaluation-sample composition under the recorded fits; it is not a
post-selection confirmation interval or proof of private leaderboard
performance.

Development Run
---------------

.. code-block:: sh

   uv run python scripts/benchmark_kaggle.py spaceship-titanic \
     --stage development \
     --recipe spaceship_inductive_v2 \
     --group-scope passenger \
     --search-profile balanced \
     --selection-strategy nested_cv \
     --selection-cv-splits 3 \
     --selection-cv-repeats 1 \
     --runs 5 \
     --n-iterations 3 \
     --exclude-models SVC MLPClassifier KNeighborsClassifier \
     --format markdown

The command writes an ignored ``latest_results.json`` and
``experiment_manifest.json`` under ``.cache/mamut/benchmark-results/``. The
manifest records configuration, data hashes, source commit, branch, dirty-tree
state, and protocol interpretation. Detailed results include the nested
selection summary, fixed-baseline comparison, and explicitly labeled
development holdout audit rows.

Locked Confirmation and Submission
----------------------------------

After choosing a candidate from development evidence, run its configuration
once on the reserved partition:

.. code-block:: sh

   uv run python scripts/benchmark_kaggle.py spaceship-titanic \
     --stage confirmation \
     --recipe spaceship_inductive_v2 \
     --group-scope passenger \
     --include-models CatBoostClassifier LGBMClassifier \
     --selection-strategy nested_cv \
     --selection-cv-splits 3 \
     --selection-cv-repeats 1 \
     --n-iterations 10 \
     --write-submission

``--write-submission`` is allowed only in the confirmation stage and produces
a local CSV. The first confirmation evaluation writes a campaign marker and
subsequent attempts are rejected, because it is no longer an unseen holdout.
Add ``--submit`` only for a frozen milestone from a clean git working tree.
The harness rejects dirty-tree uploads. Official Kaggle public
scores must be documented separately from local evidence with the manifest,
commit SHA, package version, and submission reference.

Current Evidence Status
-----------------------

The existing ``0.7843`` accuracy result is a single-run, group-disjoint
integrity smoke result from an earlier recipe and a restricted
logistic-regression/random-forest candidate pool. It is not an official Kaggle
score and not a competitive performance estimate. The v2 campaign must be run
before making performance claims.
