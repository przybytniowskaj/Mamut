import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score

import mamut.model_selection as model_selection
from mamut.wrapper import Mamut


def _exclude_all_except(model_name: str) -> list[str]:
    return [name for name in model_selection.model_param_dict if name != model_name]


PUBLIC_SKLEARN_DATASETS = [
    ("breast_cancer", load_breast_cancer),
    ("wine", load_wine),
    ("digits", load_digits),
]


@pytest.mark.parametrize(("dataset_name", "loader"), PUBLIC_SKLEARN_DATASETS)
def test_mamut_beats_dummy_baseline_on_public_sklearn_datasets(dataset_name, loader):
    X, y = loader(as_frame=True, return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    mamut = Mamut(
        exclude_models=_exclude_all_except("GaussianNB"),
        score_metric="balanced_accuracy",
        optimization_method="random_search",
        n_iterations=1,
        random_state=42,
        holdout_size=0.2,
        num_imputation="mean",
    )

    mamut.fit(X, y)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(mamut.X_train, mamut.y_train)
    baseline_score = balanced_accuracy_score(
        mamut.y_holdout,
        baseline.predict(mamut.X_holdout),
    )
    mamut_score = mamut.holdout_summary_.iloc[0]["balanced_accuracy_score"]

    assert mamut_score > baseline_score, dataset_name
