import numpy as np
import pandas as pd

from mamut.preprocessing.preprocessing import Preprocessor
from tests.mock import X, binary_y


def test_preprocessor_binary_y(X, binary_y):
    prp = Preprocessor(feature_selection=True, pca=True)
    Xft, yft = prp.fit_transform(X, binary_y)
    Xt = prp.transform(X)

    assert isinstance(Xft, np.ndarray)
    assert isinstance(yft, np.ndarray)
    assert isinstance(Xt, np.ndarray)


def test_subsequent_transform_calls(X, binary_y):
    prp = Preprocessor()
    Xft, yft = prp.fit_transform(X, binary_y)
    Xt = prp.transform(X)
    Xt2 = prp.transform(X)

    assert (Xt == Xt2).all().all()
    assert isinstance(Xt2, np.ndarray)
    assert isinstance(Xft, np.ndarray)
    assert isinstance(yft, np.ndarray)
    assert isinstance(Xt, np.ndarray)


def test_fit_transform_resets_inferred_feature_state(binary_y):
    prp = Preprocessor()
    X_first = pd.DataFrame({"num_a": np.arange(100), "cat_a": ["a", "b"] * 50})
    X_second = pd.DataFrame({"num_b": np.arange(100), "cat_b": ["x", "y"] * 50})

    prp.fit_transform(X_first, binary_y)
    prp.fit_transform(X_second, binary_y)

    assert prp.numeric_features == ["num_b"]
    assert prp.categorical_features == ["cat_b"]


def test_tree_ohe_profile_skips_numeric_scaling_and_keeps_numpy_output(X, binary_y):
    prp = Preprocessor(profile="tree_ohe")

    Xft, _ = prp.fit_transform(X, binary_y)

    assert isinstance(Xft, np.ndarray)
    assert prp.scaler_ is None
    assert "category_encoding" in prp.report()


def test_native_categorical_profile_preserves_dataframe_categories(X, binary_y):
    X = X.copy()
    X.loc[0, "cat1"] = np.nan
    prp = Preprocessor(profile="native_categorical")

    Xft, _ = prp.fit_transform(X, binary_y)
    Xt = prp.transform(X.head())

    assert isinstance(Xft, pd.DataFrame)
    assert isinstance(Xt, pd.DataFrame)
    assert str(Xft["cat1"].dtype) == "category"
    assert prp.missing_num_trans_ is None
    assert prp.scaler_ is None


def test_transform_imputes_missing_values_not_seen_during_fit():
    X_train = pd.DataFrame({"value": [1.0, 2.0, 3.0], "kind": ["a", "b", "a"]})
    X_new = pd.DataFrame({"value": [np.nan], "kind": [np.nan]})
    y_train = pd.Series([0, 1, 0])
    prp = Preprocessor(num_imputation="mean", cat_imputation="most_frequent")

    prp.fit_transform(X_train, y_train)
    transformed = prp.transform(X_new)

    assert np.isfinite(transformed).all()
