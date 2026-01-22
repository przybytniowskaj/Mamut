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
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.num_imputation = num_imputation
        self.cat_imputation = cat_imputation
        self.feature_selection = feature_selection
        self.pca = pca
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
        self.report_ = dict()

        if not self.numeric_features:
            self.numeric_features = X.select_dtypes(include="number").columns.tolist()
        if not self.categorical_features:
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
                X, self.missing_num_trans_ = handle_missing_numeric(
                    X, self.numeric_features, self.num_imputation
                )

        if self.has_categorical_:
            self.n_missing_categorical = (
                X[self.categorical_features].isnull().sum().sum()
            )
            if self.n_missing_categorical > 0:
                self.missing_categorical_ = True
                X, self.missing_cat_trans_ = handle_missing_categorical(
                    X, self.categorical_features, self.cat_imputation
                )

        self.missing_ = self.missing_numeric_ or self.missing_categorical_

        if self.missing_:
            self.report_["imputation"] = {}
            if self.missing_numeric_:
                self.report_["imputation"]["numeric"] = {
                    "transformer": self.missing_num_trans_.__class__.__name__,
                    "n_missing_numeric": self.n_missing_numeric,
                }
            if self.missing_categorical_:
                self.report_["imputation"]["categorical"] = {
                    "transformer": self.missing_cat_trans_.__class__.__name__,
                    "n_missing_categorical": self.n_missing_categorical,
                }

        if self.has_numeric_:
            n_row_before = X.shape[0]
            X, y, self.outlier_trans_ = handle_outliers(
                X, y, self.numeric_features, random_state=self.random_state
            )
            n_row_after = X.shape[0]
            self.report_["removing_outliers"] = {
                "transformer": self.outlier_trans_.__class__.__name__,
                "n_outliers_removed": n_row_before - n_row_after,
            }

        if self.has_categorical_:
            X, self.cat_trans_, self.ohe_feature_names_ = handle_categorical(
                X, self.categorical_features
            )
            self.report_["category_encoding"] = {
                "transformer": self.cat_trans_.__class__.__name__,
                "encoded_feature_names": self.ohe_feature_names_,
            }

        if self.has_numeric_:
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

        if self.has_numeric_:
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

        if self.imbalanced_resampling and self.imbalanced_:
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

        self.skewed_ = len(self.skewed_feature_names_) > 0
        self.fitted = True

        if isinstance(X, pd.DataFrame):
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
        if self.missing_numeric_:
            X[self.numeric_features] = self.missing_num_trans_.transform(
                X[self.numeric_features]
            )

        if self.missing_categorical_:
            X[self.categorical_features] = self.missing_cat_trans_.transform(
                X[self.categorical_features]
            )

        if self.has_categorical_:
            encoded_features = self.cat_trans_.transform(X[self.categorical_features])
            encoded_features_df = pd.DataFrame(
                encoded_features,
                columns=self.ohe_feature_names_,
                index=X.index,
            )
            X = X.drop(columns=self.categorical_features).join(encoded_features_df)

        if self.skewed_:
            X[self.skewed_feature_names_] = self.skew_trans_.transform(
                X[self.skewed_feature_names_]
            )

        if self.has_numeric_:
            X[self.numeric_features] = self.scaler_.transform(X[self.numeric_features])

        if self.feature_selection:
            X = self.sel_trans_.transform(X)
            X = pd.DataFrame(X, columns=self.selected_features_)

        if self.pca:
            X = self.ext_trans_.transform(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

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
