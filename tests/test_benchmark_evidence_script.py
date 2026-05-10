import pandas as pd

from scripts.benchmark_evidence import format_results


def test_benchmark_evidence_markdown_formatting():
    results = pd.DataFrame(
        [
            {
                "dataset": "example",
                "samples": 100,
                "features": 4,
                "classes": 2,
                "selected_model": "LogisticRegression",
                "holdout_score": 0.91234,
                "best_baseline": "Random Forest",
                "best_baseline_score": 0.90123,
                "repeated_cv_mean": 0.88765,
                "repeated_cv_ci": "[0.850, 0.925]",
                "guidance": "confirmed",
                "leakage_warnings": 0,
            }
        ]
    )

    output = format_results(results, "markdown")

    assert "| dataset" in output
    assert "0.912" in output
    assert "LogisticRegression" in output
