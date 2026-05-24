import warnings
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mamut.preprocessing.handlers import (
    handle_categorical,
    handle_extraction,
    handle_imbalanced,
    handle_missing_categorical,
    handle_missing_numeric,
    handle_outliers,
    handle_scaling,
    handle_selection,
    handle_skewed,
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="numpy\\.ma\\.extras",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="Conversion of an array with ndim > 0 to a scalar is deprecated.*",
)


class Preprocessor:
    """
    A class used to preprocess data for machine learning models.

    Attributes
    ----------
    numeric_features : Optional[List[str]]
        List of numeric feature names.
    categorical_features : Optional[List[str]]
        List of categorical feature names.
    num_imputation : Literal["iterative", "knn", "mean", "median", "constant"]
        Method for numeric imputation.
    cat_imputation : Literal["most_frequent", "constant"]
        Method for categorical imputation.
    scaling : Literal["standard", "robust"]
        Method for scaling numeric features.
    feature_selection : bool
        Whether to perform feature selection.
    pca : bool
        Whether to perform PCA for feature extraction.
    outlier_removal : bool
        Whether to remove rows flagged by IsolationForest during preprocessing.
    imbalanced_resampling : bool
        Whether to perform resampling to handle imbalanced data.
    resampling_strategy : Literal["SMOTE", "undersample", "combine"]
        Strategy for resampling imbalanced data.
    skew_threshold : float
        Threshold for skewness correction.
    pca_threshold : float
        Threshold for PCA feature extraction.
    selection_threshold : float
        Threshold for feature selection.
    imbalance_threshold : float
        Threshold for detecting imbalanced data.
    random_state : Optional[int]
        Random state for reproducibility.

    Methods
    -------
    fit_transform(X: pd.DataFrame, y: pd.Series) -> (np.ndarray, np.ndarray, Pipeline)
        Fits the preprocessor and transforms the data.
    transform(X: pd.DataFrame) -> np.ndarray
        Transforms the data using the fitted preprocessor.
    report() -> dict
        Returns a report of the preprocessing steps.
    _check_fitted()
        Checks if the preprocessor has been fitted.
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        num_imputation: Literal[
            "iterative", "knn", "mean", "median", "constant"
        ] = "knn",
        cat_imputation: Literal["most_frequent", "constant"] = "most_frequent",
        scaling: Literal["standard", "robust"] = "standard",
        feature_selection: bool = False,
        pca: bool = False,
        profile: Literal[
            "generic_ohe", "tree_ohe", "native_categorical"
        ] = "generic_ohe",
        outlier_removal: bool = False,
        imbalanced_resampling: bool = True,
        resampling_strategy: Literal["SMOTE", "undersample", "combine"] = "SMOTE",
        skew_threshold: float = 1,
        pca_threshold: float = 0.95,
        selection_threshold: float = 0.05,
        imbalance_threshold: float = 0.10,
        random_state: Optional[int] = 42,
    ) -> None:
        """
        Constructs all the necessary attributes for the Preprocessor object.

        Parameters
        ----------
        numeric_features : Optional[List[str]]
            List of numeric feature names.
        categorical_features : Optional[List[str]]
            List of categorical feature names.
        num_imputation : Literal["iterative", "knn", "mean", "median", "constant"]
            Method for numeric imputation.
        cat_imputation : Literal["most_frequent", "constant"]
            Method for categorical imputation.
        scaling : Literal["standard", "robust"]
            Method for scaling numeric features.
        feature_selection : bool
            Whether to perform feature selection.
        pca : bool
            Whether to perform PCA for feature extraction.
        profile : Literal["generic_ohe", "tree_ohe", "native_categorical"]
            Preprocessing family. ``generic_ohe`` keeps the legacy one-hot,
            skew-correction, and scaling path. ``tree_ohe`` one-hot encodes
            categoricals but skips numeric scaling/skew transforms.
            ``native_categorical`` preserves pandas categorical columns for
            estimators that support them.
        outlier_removal : bool
            Whether to remove rows flagged by IsolationForest. Disabled by
            default because automatic row removal can change the target
            distribution and hurt external validity.
        imbalanced_resampling : bool
            Whether to perform resampling to handle imbalanced data.
        resampling_strategy : Literal["SMOTE", "undersample", "combine"]
            Strategy for resampling imbalanced data.
        skew_threshold : float
            Threshold for skewness correction.
        pca_threshold : float
            Threshold for PCA feature extraction.
        selection_threshold : float
            Threshold for feature selection.
        imbalance_threshold : float
            Threshold for detecting imbalanced data.
        random_state : Optional[int]
            Random state for reproducibility.
        """
        self._numeric_features_config = (
            list(numeric_features) if numeric_features is not None else None
        )
        self._categorical_features_config = (
            list(categorical_features) if categorical_features is not None else None
        )
        self.numeric_features = None
        self.categorical_features = None
        self.num_imputation = num_imputation
        self.cat_imputation = cat_imputation
        if profile not in {"generic_ohe", "tree_ohe", "native_categorical"}:
            raise ValueError(
                "profile must be one of: 'generic_ohe', 'tree_ohe', "
                "'native_categorical'."
            )
        if profile == "native_categorical" and (feature_selection or pca):
            raise ValueError(
                "feature_selection and pca are not compatible with "
                "native_categorical preprocessing."
            )
        self.feature_selection = feature_selection
        self.pca = pca
        self.profile = profile
        self.outlier_removal = outlier_removal
        self.random_state = random_state
        self.scaling = scaling
        self.pca_threshold = pca_threshold
        self.selection_threshold = selection_threshold
        self.imbalance_threshold = imbalance_threshold
        self.imbalanced_resampling = imbalanced_resampling
        self.resampling_strategy = resampling_strategy
        self.skew_threshold = skew_threshold

        self.imbalanced_ = None
        self.missing_ = None
        self.imbalanced_trans_ = None
        self.outlier_trans_ = None
        self.missing_num_trans_ = None
        self.missing_cat_trans_ = None
        self.cat_trans_ = None
        self.skew_trans_ = None
        self.skewed_ = None
        self.scaler_ = None
        self.sel_trans_ = None
        self.ext_trans_ = None
        self.fitted = False
        self.skewed_feature_names_ = None
        self.selected_features_ = None
        self.pca_loadings_ = None
        self.missing_numeric_ = None
        self.missing_categorical_ = None
        self.has_numeric_ = None
        self.has_categorical_ = None
        self.ohe_feature_names_ = None
        self.report_ = None
        self.n_missing_numeric = None
        self.n_missing_categorical = None
        self.lambdas_ = None
        self.feature_importances_ = None
        self.original_feature_names_ = None
        self.feature_names_out_ = None

        self._reset_fit_state()

    @property
    def _uses_native_categorical(self) -> bool:
        return self.profile == "native_categorical"

    @property
    def _encodes_categorical(self) -> bool:
        return not self._uses_native_categorical

    @property
    def _transforms_numeric_shape(self) -> bool:
        return self.profile == "generic_ohe"

    def _reset_fit_state(self) -> None:
        self.numeric_features = (
            list(self._numeric_features_config)
            if self._numeric_features_config is not None
            else None
        )
        self.categorical_features = (
            list(self._categorical_features_config)
            if self._categorical_features_config is not None
            else None
        )
        self.imbalanced_ = False
        self.missing_ = False
        self.imbalanced_trans_ = None
        self.outlier_trans_ = None
        self.missing_num_trans_ = None
        self.missing_cat_trans_ = None
        self.cat_trans_ = None
        self.skew_trans_ = None
        self.skewed_ = False
        self.scaler_ = None
        self.sel_trans_ = None
        self.ext_trans_ = None
        self.fitted = False
        self.skewed_feature_names_ = []
        self.selected_features_ = None
        self.pca_loadings_ = None
        self.missing_numeric_ = False
        self.missing_categorical_ = False
        self.has_numeric_ = False
        self.has_categorical_ = False
        self.ohe_feature_names_ = []
        self.report_ = None
        self.n_missing_numeric = 0
        self.n_missing_categorical = 0
        self.lambdas_ = []
        self.feature_importances_ = None
        self.original_feature_names_ = None
        self.feature_names_out_ = None

    def _validate_feature_columns(self, X: pd.DataFrame) -> None:
        configured_features = []
        for feature_group in (self.numeric_features, self.categorical_features):
            if feature_group is not None:
                configured_features.extend(feature_group)

        missing_features = sorted(set(configured_features) - set(X.columns))
        if missing_features:
            raise ValueError(
                "Configured preprocessing features are not present in X: "
                f"{missing_features}."
            )

        if self.numeric_features is not None and self.categorical_features is not None:
            overlap = sorted(
                set(self.numeric_features) & set(self.categorical_features)
            )
            if overlap:
                raise ValueError(
                    "Features cannot be both numeric and categorical: " f"{overlap}."
                )

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> (np.ndarray, np.ndarray, Pipeline):
        """
        Fits the preprocessor and transforms the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series
            The target variable.

        Returns
        -------
        np.ndarray
            Transformed features.
        np.ndarray
            Transformed target variable.
        Pipeline
            The fitted pipeline.
        """
        self._reset_fit_state()
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self._validate_feature_columns(X)
        self.report_ = dict()
        self.original_feature_names_ = X.columns.tolist()

        if self.numeric_features is None:
            self.numeric_features = X.select_dtypes(include="number").columns.tolist()
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(
                exclude="number"
            ).columns.tolist()

        self.report_["features"] = {
            "numeric": self.numeric_features,
            "categorical": self.categorical_features,
        }

        if y.value_counts(normalize=True).min() < self.imbalance_threshold:
            self.imbalanced_ = True
        X, y = X.copy(), y.copy()

        self.has_numeric_ = len(self.numeric_features) > 0
        self.has_categorical_ = len(self.categorical_features) > 0

        if self.has_numeric_:
            self.n_missing_numeric = X[self.numeric_features].isnull().sum().sum()
            if self.n_missing_numeric > 0:
                self.missing_numeric_ = True
            if not self._uses_native_categorical:
                X, self.missing_num_trans_ = handle_missing_numeric(
                    X, self.numeric_features, self.num_imputation
                )

        if self.has_categorical_:
            self.n_missing_categorical = (
                X[self.categorical_features].isnull().sum().sum()
            )
            if self.n_missing_categorical > 0:
                self.missing_categorical_ = True
            cat_imputation = (
                "constant" if self._uses_native_categorical else self.cat_imputation
            )
            X, self.missing_cat_trans_ = handle_missing_categorical(
                X, self.categorical_features, cat_imputation
            )

        self.missing_ = self.missing_numeric_ or self.missing_categorical_

        if self.missing_:
            self.report_["imputation"] = {}
            if self.missing_numeric_:
                if self.missing_num_trans_ is None:
                    self.report_["imputation"]["numeric"] = {
                        "policy": "preserved_for_native_estimator",
                        "n_missing_numeric": self.n_missing_numeric,
                    }
                else:
                    self.report_["imputation"]["numeric"] = {
                        "transformer": self.missing_num_trans_.__class__.__name__,
                        "n_missing_numeric": self.n_missing_numeric,
                    }
            if self.missing_categorical_:
                self.report_["imputation"]["categorical"] = {
                    "transformer": self.missing_cat_trans_.__class__.__name__,
                    "n_missing_categorical": self.n_missing_categorical,
                }

        if self.has_numeric_ and self.outlier_removal:
            n_row_before = X.shape[0]
            X, y, self.outlier_trans_ = handle_outliers(
                X, y, self.numeric_features, random_state=self.random_state
            )
            n_row_after = X.shape[0]
            self.report_["removing_outliers"] = {
                "transformer": self.outlier_trans_.__class__.__name__,
                "n_outliers_removed": n_row_before - n_row_after,
            }

        if self.has_categorical_ and self._encodes_categorical:
            X, self.cat_trans_, self.ohe_feature_names_ = handle_categorical(
                X, self.categorical_features
            )
            self.ohe_feature_names_ = list(self.ohe_feature_names_)
            self.report_["category_encoding"] = {
                "transformer": self.cat_trans_.__class__.__name__,
                "encoded_feature_names": self.ohe_feature_names_,
            }
        elif self.has_categorical_:
            X = self._coerce_native_categorical(X)
            self.report_["category_encoding"] = {
                "transformer": "NativeCategoricalDtype",
                "categorical_feature_names": self.categorical_features,
            }
        if self._uses_native_categorical and self.has_numeric_:
            X = self._coerce_native_numeric(X)

        if self.has_numeric_ and self._transforms_numeric_shape:
            (
                X,
                self.skew_trans_,
                self.skewed_feature_names_,
                self.lambdas_,
            ) = handle_skewed(X, self.numeric_features, threshold=self.skew_threshold)
            self.report_["skew_transform"] = {
                "transformer": self.skew_trans_.__class__.__name__,
                "method": self.skew_trans_.method,
                "skewed_feature_names": self.skewed_feature_names_,
                "lambdas": self.lambdas_,
            }
        else:
            self.skewed_feature_names_ = []
            self.lambdas_ = []

        if self.has_numeric_ and self._transforms_numeric_shape:
            X, self.scaler_ = handle_scaling(X, self.numeric_features, self.scaling)
            self.report_["scaling"] = {
                "transformer": self.scaler_.__class__.__name__,
            }

        if self.feature_selection:
            X, self.sel_trans_, self.selected_features_, self.feature_importances_ = (
                handle_selection(
                    X,
                    y,
                    threshold=self.selection_threshold,
                    random_state=self.random_state,
                )
            )
            self.report_["feature_selection"] = {
                "transformer": self.sel_trans_.__class__.__name__,
                "estimator": self.sel_trans_.estimator_.__class__.__name__,
                "selected_features": self.selected_features_,
                "feature_importances": self.feature_importances_,
            }

        if self.pca:
            n_features_before = X.shape[1]
            X, self.ext_trans_, self.pca_loadings_ = handle_extraction(
                X, threshold=self.pca_threshold, random_state=self.random_state
            )
            n_features_after = X.shape[1]
            self.report_["feature_extraction"] = {
                "transformer": self.ext_trans_.__class__.__name__,
                "pca_loadings": self.pca_loadings_,
                "n_features_before": n_features_before,
                "n_features_after": n_features_after,
            }

        if all(
            [
                self.imbalanced_resampling,
                self.imbalanced_,
                not self._uses_native_categorical,
            ]
        ):
            n_row_before = X.shape[0]
            X, y, self.imbalanced_trans_ = handle_imbalanced(
                X, y, self.resampling_strategy, random_state=self.random_state
            )
            if self.imbalanced_trans_ is not None:
                n_row_after = X.shape[0]
                self.report_["imbalanced_resampling"] = {
                    "transformer": self.imbalanced_trans_.__class__.__name__,
                    "n_resampled": n_row_after - n_row_before,
                }
            else:
                self.report_["imbalanced_resampling"] = {
                    "skipped": True,
                    "reason": "Insufficient minority samples for SMOTE.",
                    "strategy": self.resampling_strategy,
                }
        elif self.imbalanced_resampling and self.imbalanced_:
            self.report_["imbalanced_resampling"] = {
                "skipped": True,
                "reason": "Native categorical preprocessing preserves raw feature types.",
                "strategy": self.resampling_strategy,
            }

        self.skewed_ = len(self.skewed_feature_names_) > 0
        if self.pca:
            self.feature_names_out_ = [
                f"PC{i + 1}" for i in range(self.pca_loadings_.shape[0])
            ]
        elif isinstance(X, pd.DataFrame):
            self.feature_names_out_ = X.columns.tolist()
        else:
            self.feature_names_out_ = [
                f"feature_{i}" for i in range(np.asarray(X).shape[1])
            ]
        self.fitted = True

        if isinstance(X, pd.DataFrame) and not self._uses_native_categorical:
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        return X, y

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the data using the fitted preprocessor.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.

        Returns
        -------
        np.ndarray
            Transformed features.
        """
        self._check_fitted()
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        X = X.copy()
        if self.missing_num_trans_ is not None:
            X[self.numeric_features] = self.missing_num_trans_.transform(
                X[self.numeric_features]
            )

        if self.missing_cat_trans_ is not None:
            X[self.categorical_features] = self.missing_cat_trans_.transform(
                X[self.categorical_features]
            )

        if self.has_categorical_ and self._encodes_categorical:
            encoded_features = self.cat_trans_.transform(X[self.categorical_features])
            encoded_features_df = pd.DataFrame(
                encoded_features,
                columns=self.ohe_feature_names_,
                index=X.index,
            )
            X = X.drop(columns=self.categorical_features).join(encoded_features_df)
        elif self.has_categorical_:
            X = self._coerce_native_categorical(X)
        if self._uses_native_categorical and self.has_numeric_:
            X = self._coerce_native_numeric(X)

        if self.skewed_:
            X[self.skewed_feature_names_] = self.skew_trans_.transform(
                X[self.skewed_feature_names_]
            )

        if self.has_numeric_ and self.scaler_ is not None:
            X[self.numeric_features] = self.scaler_.transform(X[self.numeric_features])

        if self.feature_selection:
            index = X.index
            selected = self.sel_trans_.transform(X)
            X = pd.DataFrame(selected, columns=self.selected_features_, index=index)

        if self.pca:
            X = self.ext_trans_.transform(X)

        if isinstance(X, pd.DataFrame) and not self._uses_native_categorical:
            X = X.values

        return X

    def _coerce_native_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.categorical_features:
            X[feature] = (
                X[feature].astype("string").fillna("__missing__").astype("category")
            )
        return X

    def _coerce_native_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.numeric_features:
            X[feature] = pd.to_numeric(X[feature], errors="coerce")
        return X

    def report(self):
        """
        Returns a report of the preprocessing steps.

        Returns
        -------
        dict
            A dictionary containing the report of the preprocessing steps.
        """
        self._check_fitted()
        return self.report_

    def _check_fitted(self):
        """
        Checks if the preprocessor has been fitted.

        Raises
        ------
        RuntimeError
            If the preprocessor has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fitted.")
