Dependency Update Policy
========================

.. meta::
   :description: Dependency update policy for MAMUT, including uv lock handling, ML/runtime validation, and automation rules.
   :keywords: MAMUT dependency policy, uv lock, Dependabot, ML dependencies

MAMUT uses ``uv`` for dependency locking and installation. Keep dependency
changes synchronized between ``pyproject.toml`` and ``uv.lock`` by using uv
commands instead of editing the lockfile manually.

Routine Updates
---------------

Patch updates can be normal pull requests when they do not change public
behavior. Validate them with:

.. code-block:: sh

   uv sync --locked --all-groups
   uv run deptry .
   scripts/audit_dependencies.sh
   uv run pytest
   uv run pre-commit run --all-files

Tooling Updates
---------------

Development, documentation, packaging, and security tooling can be grouped when
the change is limited to minor or patch versions. Examples include ``pytest``,
``pre-commit``, ``deptry``, ``sphinx``, ``ipykernel``, ``twine``, and
``pip-audit``.

ML and Runtime Updates
----------------------

Minor updates to ML/runtime dependencies require full validation because they
can change model behavior, numerical output, or generated reports. Treat
``numpy``, ``scikit-learn``, ``xgboost``, ``shap``, ``imbalanced-learn``,
``pandas``, ``scipy``, ``lightgbm``, and ``catboost`` as behavior-sensitive
dependencies.

For these updates, run the routine checks plus:

.. code-block:: sh

   uv build
   uv run twine check dist/*
   uv run make -C docs html
   uv run sphinx-build -W --keep-going -b html docs/source docs/build/html-strict
   uv run python scripts/benchmark_evidence.py --format markdown

Documentation validation requires the system ``pandoc`` executable because the
published walkthrough is rendered through ``nbsphinx``.

Review model-selection behavior, preprocessing outputs, and report artifacts
before merging.

Major Updates
-------------

Major runtime or ML updates must use dedicated pull requests. Include release
notes or changelog links, explain expected behavior changes, and keep unrelated
dependency updates out of the same PR.

Automation
----------

Dependabot opens low-volume update PRs for ``uv`` and GitHub Actions. The
dependency health workflow is read-only: it reports outdated packages and runs a
dry-run lock resolution, but it does not commit upgrades.
