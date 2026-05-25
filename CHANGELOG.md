# Changelog

All notable changes to MAMUT are documented here.

## Unreleased

## [0.3.0] - 2026-05-25

### Added

- Added model-aware candidate selection with configurable search profiles and explicit LightGBM and CatBoost support.
- Added grouped and nested selection diagnostics, evidence-guided challenges, and fixed-parameter refitting for benchmark submissions.
- Added a reproducible Spaceship Titanic benchmark harness with immutable campaign manifests, locked confirmation handling, relational-overlap audits, and bounded-runtime controls.
- Added focused tests for model registries, benchmark protocol behavior, public prediction contracts, and candidate-specific preprocessing.

### Changed

- Expanded the documentation around validation estimands, model-selection integrity, Kaggle evidence, and honest interpretation of leaderboard observations.
- Recorded official Spaceship Titanic public scores of `0.79798` and `0.80617`; the higher score followed observation of an earlier submission and is reported as post-leaderboard development evidence, not an independent final estimate.
- Updated development tooling with `deptry` and `pytest` maintenance releases while retaining the validated ML/runtime dependency set for this release.

### Fixed

- Prevented submission generation from starting a fresh tuning search after locked confirmation by refitting the confirmed hyperparameters on all labelled training data.
- Accelerated group-bootstrap benchmark summaries and documented candidate-fit/runtime budgets for practical experiment control.
- Corrected generated benchmark documentation so displayed baseline challenges match the locked release environment.

## [0.2.1] - 2026-05-12

### Fixed

- Made mixed numeric/categorical report generation robust to categorical columns and missing numeric values.
- Made public model predictions return original target labels instead of internal encoded labels.
- Reset preprocessing fit state between fits and tolerate unseen categorical levels at prediction time.
- Added explicit validation for model-search configuration.

### Added

- Added configurable evaluation outputs for custom report directories, optional SHAP, optional HTML writing, and optional plot artifacts.
- Added an opt-in final refit path for the selected estimator on all non-holdout modeling data.
- Added generated documentation sitemap and robots metadata for Read the Docs.

### Changed

- Improved README, quickstart, report, and user-guide documentation around holdout evaluation, lightweight reports, final refit behavior, and approximate evidence intervals.
- Updated Sphinx configuration for canonical Read the Docs URLs, public-page sitemap output, and stricter documentation hygiene.

## [0.2.0] - 2026-05-09

### Added

- Evidence reporting with validation integrity, leakage checks, baseline comparison, repeated stratified cross-validation, confidence intervals, and evidence-guided selection guidance.
- Optional final holdout evaluation so final report scores can be separated from model and ensemble selection.
- Public package import via `from mamut import Mamut` and package `__version__`.
- Reproducible evidence benchmark script for lightweight sklearn dataset diagnostics.
- Dependency update policy, Dependabot configuration, dependency health workflow, and scheduled security audit.

### Changed

- Migrated local development and CI workflows from Poetry/requirements files to `uv`.
- Made fitted model artifact writing opt-in with `save_models=True`.
- Reworked documentation for Read the Docs, stricter Sphinx builds, and clearer package positioning.
- Clarified validation semantics across code, reports, tests, README, and user documentation.

### Validation

- Release gate covers dependency declaration checks, vulnerability audit, pytest, pre-commit, strict docs build, linkcheck, package build, metadata validation, and built-wheel smoke testing.

## [0.1.2] and Earlier

Earlier releases established the core automated tabular classification workflow, preprocessing pipeline, Optuna-based model search, HTML reports, SHAP output, and initial packaging.
