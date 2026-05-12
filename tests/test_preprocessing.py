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
