import numpy as np
import pandas as pd
import pytest

from mamut.wrapper import Mamut
from tests.mock import X, X_missing, binary_y, imbalanced_y, multiclass_y


def test_wrapper_binary_target(X, binary_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X, binary_y)
    pred = mamut.best_model_.predict(X)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1


def test_wrapper_multiclass_target(X, multiclass_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X, multiclass_y)
    pred = mamut.best_model_.predict(X)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1


def test_wrapper_imbalanced_target(X, imbalanced_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X, imbalanced_y)
    pred = mamut.best_model_.predict(X)

    n_before = X.shape[0]
    n_after = mamut.X_train.shape[0] + mamut.X_validation.shape[0]

    assert n_after > n_before
    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_missing_data(X_missing, binary_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X_missing, binary_y)
    pred = mamut.best_model_.predict(X_missing)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_pca(X, binary_y):
    mamut = Mamut(n_iterations=1, pca=True)
    mamut.fit(X, binary_y)
    pred = mamut.best_model_.predict(X)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_selection(X, binary_y):
    mamut = Mamut(n_iterations=1, feature_selection=True)
    mamut.fit(X, binary_y)
    pred = mamut.best_model_.predict(X)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_subsequent_predict_calls(X, binary_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X, binary_y)
    pred = mamut.best_model_.predict(X)
    pred2 = mamut.best_model_.predict(X)

    assert (pred == pred2).all()
    assert isinstance(pred, np.ndarray)
    assert isinstance(pred2, np.ndarray)
    assert mamut.best_score_ is not None


def test_wrapper_holdout_split_is_available_for_final_evaluation(X, binary_y):
    mamut = Mamut(n_iterations=1, holdout_size=0.2)
    mamut.fit(X, binary_y)

    assert mamut.X_holdout is not None
    assert mamut.y_holdout is not None
    assert mamut.holdout_summary_ is not None
    assert mamut.holdout_score_ is not None
    assert len(mamut.y_holdout) == 20
    assert len(mamut.y_validation) == 16


def test_wrapper_explicit_holdout_is_not_used_as_validation(X, binary_y):
    holdout_idx = list(range(40, 50)) + list(range(90, 100))
    modeling_idx = [idx for idx in range(len(X)) if idx not in holdout_idx]
    X_modeling = X.iloc[modeling_idx].copy()
    y_modeling = binary_y.iloc[modeling_idx].copy()
    X_holdout = X.iloc[holdout_idx].copy()
    y_holdout = binary_y.iloc[holdout_idx].copy()

    mamut = Mamut(n_iterations=1)
    mamut.fit(X_modeling, y_modeling, X_holdout=X_holdout, y_holdout=y_holdout)

    assert len(mamut.y_holdout) == 20
    assert len(mamut.y_validation) == 16
    assert mamut.X_holdout_raw_.equals(X_holdout)


def test_wrapper_grouped_holdout_and_validation_are_group_disjoint(X, binary_y):
    groups = np.repeat(np.arange(50), 2)
    mamut = Mamut(n_iterations=1, holdout_size=0.2)

    mamut.fit(X, binary_y, groups=groups)

    assert set(mamut.groups_modeling_).isdisjoint(mamut.groups_holdout_)
    assert set(mamut.groups_train_).isdisjoint(mamut.groups_validation_)


def test_wrapper_rejects_overlapping_explicit_holdout_groups(X, binary_y):
    mamut = Mamut(n_iterations=1)

    with pytest.raises(ValueError, match="overlap modeling groups"):
        mamut.fit(
            X.iloc[:80],
            binary_y.iloc[:80],
            X_holdout=X.iloc[80:],
            y_holdout=binary_y.iloc[80:],
            groups=pd.Series(np.arange(80)),
            groups_holdout=pd.Series([79, *range(81, 100)]),
        )


def test_wrapper_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="score_metric"):
        Mamut(score_metric="bad")
    with pytest.raises(ValueError, match="optimization_method"):
        Mamut(optimization_method="grid")
    with pytest.raises(ValueError, match="n_iterations"):
        Mamut(n_iterations=0)
    with pytest.raises(ValueError, match="unsupported model"):
        Mamut(exclude_models=["NoSuchModel"])
    with pytest.raises(ValueError, match="include_models"):
        Mamut(include_models=["GaussianNB"], exclude_models=["LogisticRegression"])
    with pytest.raises(ValueError, match="search_profile"):
        Mamut(search_profile="huge")
    with pytest.raises(ValueError, match="n_jobs"):
        Mamut(n_jobs=0)
    with pytest.raises(ValueError, match="selection_strategy"):
        Mamut(selection_strategy="holdout")
    with pytest.raises(ValueError, match="selection_cv_splits"):
        Mamut(selection_cv_splits=1)
    with pytest.raises(ValueError, match="selection_cv_repeats"):
        Mamut(selection_cv_repeats=0)
    with pytest.raises(ValueError, match="selection_practical_margin"):
        Mamut(selection_practical_margin=-0.1)
    with pytest.raises(ValueError, match="preprocessing_profile"):
        Mamut(preprocessing_profile="native")
