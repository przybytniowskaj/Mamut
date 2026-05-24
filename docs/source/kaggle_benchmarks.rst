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

Protocol and Estimands
----------------------

The ``development`` stage reserves a deterministic confirmation partition and
excludes it from model development within a campaign. Only a frozen candidate
may be run once with ``--stage confirmation``. A confirmation score is an
observation of that candidate; it must not select another model.

Two evaluation estimands are deliberately available:

* ``--recipe spaceship_competition_v3 --group-scope passenger`` estimates the
  Kaggle task. Exact passenger groups remain disjoint, but surname categories
  may recur across folds because the official Kaggle train/test files share
  surnames extensively. Features such as batch family size remain target-free.
* ``--recipe spaceship_cohort_v2 --group-scope household_component`` estimates
  performance for entirely unseen surname-linked components. It is stricter
  and should be used for deployment-oriented generalization claims.

The earlier ``spaceship_inductive_v2`` recipe derives row-level domain
features and surname category only; it omits the batch relational features
available in the competition-aligned recipe. Older recipes remain available
for reproducibility.

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
     --campaign-id spaceship-competition-v3 \
     --recipe spaceship_competition_v3 \
     --group-scope passenger \
     --include-models LGBMClassifier \
     --selection-strategy single_split \
     --runs 5 \
     --n-iterations 3 \
     --max-runtime-seconds 900 \
     --format markdown

Each command writes immutable ``results.json`` and ``manifest.json`` files
under ``.cache/mamut/benchmark-results/<competition>/<campaign>/<recipe>/<run>/``.
The manifest records configuration, data hashes, source commit, branch,
dirty-tree state, evaluation estimand, and train/test relational-overlap
audit. ``--max-runtime-seconds`` is a soft budget checked between completed
outer runs; a single model fit can exceed it.

Locked Confirmation and Submission
----------------------------------

After choosing a candidate from development evidence, run its configuration
once on the reserved partition:

.. code-block:: sh

   uv run python scripts/benchmark_kaggle.py spaceship-titanic \
     --stage confirmation \
     --campaign-id spaceship-competition-v3 \
     --recipe spaceship_competition_v3 \
     --group-scope passenger \
     --include-models CatBoostClassifier \
     --selection-strategy single_split \
     --n-iterations 5 \
     --write-submission

``--write-submission`` is allowed only in the confirmation stage and produces
a local CSV. The first confirmation evaluation writes a marker inside that
campaign and subsequent confirmation attempts in the same campaign are
rejected. Add ``--submit`` only for a frozen milestone from a clean git
working tree. After any leaderboard result is observed, a later campaign is
useful for iteration but is not an independent final performance test.
For new submissions, the harness records the hyperparameters chosen without
confirmation labels and refits that fixed candidate on all labeled training
rows; it does not start a fresh tuning search while generating the CSV.

Current Evidence Status
-----------------------

The first official MAMUT submission used commit ``ca29cc8``,
``spaceship_inductive_v2``, and ``LGBMClassifier``. Five development folds
averaged ``0.8064`` accuracy; the locked confirmation score was ``0.7967``;
the Kaggle public score was ``0.79798`` (submission ``52996032``).

A post-leaderboard development campaign at commit ``1d4dd1c`` identified
CatBoost as stronger than LightGBM on paired outer folds using
``spaceship_inductive_v2`` (``0.8094`` versus ``0.8016`` mean accuracy).
Its confirmation observation was ``0.8035`` and the resulting public Kaggle
score was ``0.80617`` (submission ``52996964``), exceeding the repository
owner's earlier ``cat.csv`` score of ``0.80500``. Because this iteration
followed observation of a public score, it is useful evidence of improvement,
not an independent final estimate. That submission predates the fixed-parameter
refit rule described above; future submission artifacts record and refit the
confirmation-selected hyperparameters explicitly.

Audit interpretation: the official train/test files share surname values for
``87.9%`` of test rows but do not share exact passenger-group or cabin keys.
The submitted recipe therefore used a legitimate competition-aligned surname
signal, but did not exploit target-free family/group batch aggregates and did
did not establish superior competitive performance. In paired post-score
development evaluation, adding those batch aggregates improved LightGBM only
slightly and did not improve CatBoost; model-family choice accounted for the
substantial measured gain.
