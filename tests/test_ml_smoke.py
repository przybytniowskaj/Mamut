import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

import mamut.model_selection as model_selection
import mamut.wrapper as wrapper_module
from mamut.evaluation import ModelEvaluator
from mamut.wrapper import Mamut

SMALL_XGBOOST_SEARCH = {
    "n_estimators": (3, 5, "int"),
    "learning_rate": (0.10, 0.30, "float"),
    "subsample": (0.8, 1.0, "float"),
    "booster": (["gbtree"], "categorical"),
    "max_depth": (1, 2, "int"),
    "min_child_weight": (1.0, 2.0, "float"),
    "colsample_bytree": (0.8, 1.0, "float"),
    "colsample_bylevel": (0.8, 1.0, "float"),
    "reg_alpha": (1e-4, 1e-3, "log"),
    "reg_lambda": (1.0, 2.0, "float"),
    "tree_method": (["hist"], "categorical"),
    "n_jobs": ([1], "categorical"),
    "verbosity": ([0], "categorical"),
}


def _exclude_all_except(model_name: str) -> list[str]:
    return [name for name in model_selection.model_param_dict if name != model_name]


def _numeric_classification_frame(
    n_samples: int = 80,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    target = np.tile([0, 1], n_samples // 2)
    features = pd.DataFrame(
        {
            "signal": target + rng.normal(0, 0.15, n_samples),
            "margin": (target * 2 - 1) + rng.normal(0, 0.25, n_samples),
            "noise_a": rng.normal(0, 1, n_samples),
            "noise_b": rng.normal(0, 1, n_samples),
        }
    )
    order = rng.permutation(n_samples)

    return (
        features.iloc[order].reset_index(drop=True),
        pd.Series(target[order], name="target"),
    )


def _mixed_classification_frame(n_samples: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    X, y = _numeric_classification_frame(n_samples)
    X = X[["signal", "noise_a"]].copy()
    X["segment"] = np.where(y.to_numpy() == 1, "high", "low")
    X["region"] = np.where(np.arange(n_samples) % 2 == 0, "north", "south")
    X.loc[[1, 11, 21], "signal"] = np.nan
    X.loc[[5, 15, 25], "segment"] = np.nan

    return X, pd.Series(np.where(y.to_numpy() == 1, "yes", "no"), name="target")


def test_mamut_fit_predict_with_preprocessing_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _mixed_classification_frame()
    mamut = Mamut(
        exclude_models=_exclude_all_except("GaussianNB"),
        n_iterations=1,
        optimization_method="random_search",
        random_state=42,
        num_imputation="mean",
        cat_imputation="most_frequent",
    )

    fitted_model = mamut.fit(X, y)
    predictions = mamut.predict(X.head(10))
    probabilities = mamut.predict_proba(X.head(10))
    preprocessing_report = mamut.preprocessor.report()

    assert fitted_model is mamut.best_model_
    assert predictions.shape == (10,)
    assert set(predictions).issubset({"yes", "no"})
    assert probabilities.shape == (10, 2)
    assert {"imputation", "category_encoding", "scaling"}.issubset(preprocessing_report)
    assert not (tmp_path / "fitted_models").exists()


def test_mamut_fit_can_save_model_artifacts_when_requested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _mixed_classification_frame()
    mamut = Mamut(
        exclude_models=_exclude_all_except("GaussianNB"),
        n_iterations=1,
        optimization_method="random_search",
        random_state=42,
        save_models=True,
        num_imputation="mean",
        cat_imputation="most_frequent",
    )

    mamut.fit(X, y)

    assert mamut.models_output_path_ is not None
    assert (tmp_path / "fitted_models").is_dir()
    assert list((tmp_path / "fitted_models").glob("*/GaussianNB.joblib"))


def test_mamut_xgboost_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        model_selection.model_param_dict,
        "XGBClassifier",
        SMALL_XGBOOST_SEARCH,
    )
    X, y = _numeric_classification_frame()
    mamut = Mamut(
        exclude_models=_exclude_all_except("XGBClassifier"),
        n_iterations=1,
        optimization_method="random_search",
        random_state=42,
    )

    mamut.fit(X, y)
    predictions = mamut.predict(X.head(8))

    assert "XGBClassifier" in mamut.raw_fitted_models_
    assert predictions.shape == (8,)


def test_mamut_evaluate_uses_holdout_when_available(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _numeric_classification_frame()
    mamut = Mamut(
        exclude_models=_exclude_all_except("GaussianNB"),
        n_iterations=1,
        optimization_method="random_search",
        random_state=42,
        holdout_size=0.2,
    )
    mamut.fit(X, y)
    captured = {}

    class CapturingEvaluator:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def evaluate_to_html(self, summary):
            captured["summary"] = summary

        def plot_results_in_notebook(self):
            captured["plotted"] = True

    monkeypatch.setattr(wrapper_module, "ModelEvaluator", CapturingEvaluator)

    mamut.evaluate()

    assert captured["evaluation_dataset"] == "holdout"
    assert captured["rank_by_metric"] is False
    assert captured["training_summary"].equals(mamut.holdout_summary_)
    assert captured["summary"].equals(mamut.holdout_summary_)


def test_mamut_evaluate_generates_report_and_shap_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _numeric_classification_frame(n_samples=20)
    model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(
        X.to_numpy(), y.to_numpy()
    )
    predictions = model.predict(X.to_numpy())
    probabilities = model.predict_proba(X.to_numpy())[:, 1]
    training_summary = pd.DataFrame(
        [
            {
                "model": "DecisionTreeClassifier",
                "accuracy_score": accuracy_score(y, predictions),
                "balanced_accuracy_score": balanced_accuracy_score(y, predictions),
                "precision_score": precision_score(y, predictions, average="weighted"),
                "recall_score": recall_score(y, predictions, average="weighted"),
                "f1_score": f1_score(y, predictions, average="weighted"),
                "jaccard_score": jaccard_score(y, predictions, average="weighted"),
                "roc_auc_score": roc_auc_score(y, probabilities),
                "duration": 0.01,
            }
        ]
    )
    evaluator = ModelEvaluator(
        {"DecisionTreeClassifier": model},
        X_evaluation=X.to_numpy(),
        y_evaluation=y,
        X_train=X.to_numpy(),
        y_train=y.to_numpy(),
        X=X,
        y=y,
        optimizer="random_search",
        n_trials=1,
        metric="f1_score",
        studies={},
        training_summary=training_summary,
        pca_loadings=None,
        binary=True,
        preprocessing_steps={
            "features": {"numeric": X.columns.to_list(), "categorical": []}
        },
        is_ensemble=False,
        greedy_ensemble=None,
        excluded_models=[],
        n_top_models=1,
    )

    evaluator.evaluate_to_html(training_summary)

    report_dir = tmp_path / "mamut_report"
    plot_dir = report_dir / "plots"
    shap_plots = list(plot_dir.glob("shap_values.png")) + list(
        plot_dir.glob("shap_beeswarm_class_*.png")
    )

    assert list(report_dir.glob("report_*.html"))
    assert (plot_dir / "roc_auc_curve.png").is_file()
    assert (plot_dir / "confusion_matrices.png").is_file()
    assert (plot_dir / "feature_importance.png").is_file()
    assert shap_plots
    assert all(path.stat().st_size > 0 for path in shap_plots)
