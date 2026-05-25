#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

from mamut import Mamut

DatasetLoader = Callable[..., tuple[pd.DataFrame, pd.Series]]

DATASET_LOADERS: dict[str, DatasetLoader] = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}


def run_benchmark(
    dataset_names: Iterable[str],
    *,
    random_state: int = 42,
    n_iterations: int = 1,
    holdout_size: float = 0.2,
    evidence_cv_splits: int = 3,
    evidence_cv_repeats: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    rows = [
        _run_dataset(
            dataset_name,
            random_state=random_state,
            n_iterations=n_iterations,
            holdout_size=holdout_size,
            evidence_cv_splits=evidence_cv_splits,
            evidence_cv_repeats=evidence_cv_repeats,
            verbose=verbose,
        )
        for dataset_name in dataset_names
    ]
    return pd.DataFrame(rows)


def _run_dataset(
    dataset_name: str,
    *,
    random_state: int,
    n_iterations: int,
    holdout_size: float,
    evidence_cv_splits: int,
    evidence_cv_repeats: int,
    verbose: bool,
) -> dict:
    X, y = _load_dataset(dataset_name)
    mamut = Mamut(
        search_profile="quick",
        score_metric="balanced_accuracy",
        optimization_method="random_search",
        n_iterations=n_iterations,
        random_state=random_state,
        holdout_size=holdout_size,
        refit_final_model=True,
        evidence_cv_splits=evidence_cv_splits,
        evidence_cv_repeats=evidence_cv_repeats,
        num_imputation="mean",
    )

    with _maybe_quiet(verbose):
        mamut.fit(X, y)
        mamut.generate_evidence()

    guidance = mamut.selection_guidance_.iloc[0]
    selected_model = _short_selected_model_name(guidance["selected_model"])
    best_baseline = _best_baseline_row(
        mamut.baseline_comparison_, selected_model_label=guidance["selected_model"]
    )
    selected_stability = _row_for_model(
        mamut.score_stability_, guidance["selected_model"]
    )
    validation_integrity = mamut.validation_integrity_.iloc[0]

    return {
        "dataset": dataset_name,
        "samples": len(X),
        "features": X.shape[1],
        "classes": int(pd.Series(y).nunique()),
        "selected_model": selected_model,
        "holdout_score": _safe_float(guidance["selected_split_score"]),
        "best_baseline": best_baseline["model"],
        "best_baseline_score": _safe_float(best_baseline["score"]),
        "repeated_cv_mean": _safe_float(guidance["selected_stability_mean"]),
        "repeated_cv_ci": _format_ci(selected_stability),
        "guidance": guidance["status"],
        "leakage_warnings": int(validation_integrity["n_leakage_warnings"]),
    }


def _load_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.Series]:
    if dataset_name not in DATASET_LOADERS:
        valid_names = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {valid_names}."
        )

    X, y = DATASET_LOADERS[dataset_name](as_frame=True, return_X_y=True)
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y, name="target").reset_index(drop=True)
    return X, y


def _best_baseline_row(
    baseline_comparison: pd.DataFrame, *, selected_model_label: str
) -> pd.Series:
    baselines = baseline_comparison.loc[
        ~baseline_comparison["model"].str.startswith("MAMUT ")
    ].dropna(subset=["score"])
    if baselines.empty:
        return pd.Series({"model": "", "score": np.nan})
    return baselines.sort_values("score", ascending=False).iloc[0]


def _row_for_model(table: pd.DataFrame, model_name: str) -> pd.Series:
    match = table.loc[table["model"] == model_name]
    if match.empty:
        return pd.Series(dtype="object")
    return match.iloc[0]


def _short_selected_model_name(selected_model_label: str) -> str:
    prefix = "MAMUT Selected ("
    if selected_model_label.startswith(prefix) and selected_model_label.endswith(")"):
        return selected_model_label[len(prefix) : -1]
    return selected_model_label


def _safe_float(value) -> float:
    if pd.isna(value):
        return np.nan
    return float(value)


def _format_ci(row: pd.Series) -> str:
    if row.empty or pd.isna(row.get("ci_low")) or pd.isna(row.get("ci_high")):
        return "n/a"
    return f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


@contextmanager
def _maybe_quiet(verbose: bool):
    if verbose:
        yield
    else:
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                yield
        finally:
            logging.disable(previous_disable_level)


def format_results(results: pd.DataFrame, output_format: str) -> str:
    if output_format == "json":
        return results.to_json(orient="records", indent=2)
    if output_format == "csv":
        return results.to_csv(index=False, float_format="%.6f")
    if output_format == "markdown":
        return _to_markdown(_display_frame(results))
    raise ValueError(f"Unsupported output format: {output_format}")


def _display_frame(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    for column in ("holdout_score", "best_baseline_score", "repeated_cv_mean"):
        display[column] = display[column].map(_format_score)
    return display


def _format_score(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def _to_markdown(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(str(header)), max((len(row[index]) for row in rows), default=0))
        for index, header in enumerate(headers)
    ]
    header_line = _format_markdown_row([str(header) for header in headers], widths)
    separator_line = _format_markdown_row(["-" * width for width in widths], widths)
    row_lines = [_format_markdown_row(row, widths) for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_markdown_row(values: list[str], widths: list[int]) -> str:
    padded_values = [value.ljust(widths[index]) for index, value in enumerate(values)]
    return f"| {' | '.join(padded_values)} |"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAMUT's lightweight evidence benchmark on sklearn datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_LOADERS),
        default=sorted(DATASET_LOADERS),
        help="Datasets to include.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iterations", type=int, default=1)
    parser.add_argument("--holdout-size", type=float, default=0.2)
    parser.add_argument("--evidence-cv-splits", type=int, default=3)
    parser.add_argument("--evidence-cv-repeats", type=int, default=1)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show estimator optimization progress.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_benchmark(
        args.datasets,
        random_state=args.random_state,
        n_iterations=args.n_iterations,
        holdout_size=args.holdout_size,
        evidence_cv_splits=args.evidence_cv_splits,
        evidence_cv_repeats=args.evidence_cv_repeats,
        verbose=args.verbose,
    )
    print(format_results(results, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
