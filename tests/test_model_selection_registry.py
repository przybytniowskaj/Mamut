import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mamut.model_selection import (
    ModelSelector,
    available_model_names,
    preprocessing_profile_for_model,
    select_from_repeated_cv_summary,
)
from mamut.utils.utils import (
    adjust_search_spaces,
    model_names_for_profile,
    model_param_dict,
)


def test_search_profiles_select_expected_candidate_sets():
    assert model_names_for_profile("quick") == (
        "LogisticRegression",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GaussianNB",
    )
    assert "CatBoostClassifier" in model_names_for_profile("balanced")
    assert set(model_names_for_profile("thorough")) == set(available_model_names())


def test_random_forest_search_space_uses_leaf_counts_not_large_fractions():
    low, high, distribution = model_param_dict["RandomForestClassifier"][
        "min_samples_leaf"
    ]

    assert distribution == "int"
    assert low == 1
    assert high <= 20


def test_logistic_non_elasticnet_configuration_omits_l1_ratio():
    configured = adjust_search_spaces(
        {"solver": "lbfgs", "l1_ratio": 0.5},
        LogisticRegression(),
    )

    assert configured["penalty"] == "l2"
    assert "l1_ratio" not in configured


def test_boosted_native_models_use_native_categorical_preprocessing_profiles():
    assert preprocessing_profile_for_model("CatBoostClassifier") == "native_categorical"
    assert preprocessing_profile_for_model("LGBMClassifier") == "native_categorical"
    assert preprocessing_profile_for_model("RandomForestClassifier") == "tree_ohe"


def test_include_models_takes_precedence_over_search_profile():
    selected = ModelSelector._resolve_model_names(
        include_models=["KNeighborsClassifier"],
        exclude_models=None,
        search_profile="quick",
    )

    assert selected == ["KNeighborsClassifier"]


def test_exclude_models_preserves_backward_compatible_all_model_pool():
    selected = ModelSelector._resolve_model_names(
        include_models=None,
        exclude_models=[
            name for name in available_model_names() if name != "KNeighborsClassifier"
        ],
        search_profile="quick",
    )

    assert selected == ["KNeighborsClassifier"]


def test_model_selector_fits_preprocessing_inside_each_cv_fold(monkeypatch):
    instances = []

    class RecordingPreprocessor:
        def __init__(self):
            self.fit_index = None
            self.transform_index = None
            instances.append(self)

        def fit_transform(self, X, y):
            self.fit_index = set(X.index)
            return X[["signal"]].to_numpy(), np.asarray(y)

        def transform(self, X):
            self.transform_index = set(X.index)
            return X[["signal"]].to_numpy()

    X = pd.DataFrame({"signal": np.arange(20), "noise": np.arange(20) % 3})
    y = pd.Series([0, 1] * 10)
    monkeypatch.setitem(
        model_param_dict,
        "GaussianNB",
        {"var_smoothing": (1e-9, 1e-9, "log")},
    )
    selector = ModelSelector(
        X.to_numpy(),
        y.to_numpy(),
        X.to_numpy(),
        y.to_numpy(),
        score_metric=lambda y_true, y_pred: float(np.mean(y_true == y_pred)),
        X_train_raw=X,
        y_train_raw=y,
        preprocessor_factory=RecordingPreprocessor,
        include_models=["GaussianNB"],
        optimization_method="random_search",
        n_iterations=1,
        random_state=42,
    )

    selector.optimize_model(selector.models[0])

    assert len(instances) == 5
    assert all(
        instance.fit_index.isdisjoint(instance.transform_index)
        for instance in instances
    )


def test_repeated_cv_selection_prefers_lower_variance_inside_practical_tie():
    summary = pd.DataFrame(
        [
            {
                "model": "HighMean",
                "mean_score": 0.812,
                "std_score": 0.040,
                "selection_duration": 1.0,
                "status": "ok",
            },
            {
                "model": "StableTie",
                "mean_score": 0.810,
                "std_score": 0.010,
                "selection_duration": 2.0,
                "status": "ok",
            },
        ]
    )

    selected = select_from_repeated_cv_summary(summary, practical_margin=0.005)

    assert selected == "StableTie"
