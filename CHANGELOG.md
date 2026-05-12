# Changelog

All notable changes to MAMUT are documented here.

## Unreleased

### Fixed

- Made mixed numeric/categorical report generation robust to categorical columns and missing numeric values.
- Made public model predictions return original target labels instead of internal encoded labels.
- Reset preprocessing fit state between fits and tolerate unseen categorical levels at prediction time.
- Added explicit validation for model-search configuration.

### Added

- Added configurable evaluation outputs for custom report directories, optional SHAP, optional HTML writing, and optional plot artifacts.
- Added an opt-in final refit path for the selected estimator on all non-holdout modeling data.

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
