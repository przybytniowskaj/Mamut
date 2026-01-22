from typing import List, Literal

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

import mamut.preprocessing.settings as settings


def handle_outliers(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> (pd.DataFrame, pd.Series, IsolationForest):
    """
    Handles outliers in the dataset using IsolationForest.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        y: pd.Series
            Target array.
        feature_names: List[str]
            Names of the features in the dataset.
        contamination: float
            The proportion of outliers in the data.
        random_state: int
            Seed for reproducibility.

    Returns:
        X_filtered: pd.DataFrame
            Feature matrix with outliers removed.
        y_filtered: pd.Series
            Target array with outliers removed.
        transformer: IsolationForest
            Fitted IsolationForest model.
    """
    X, y = X.copy(), y.copy()
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = iso_forest.fit_predict(X[feature_names])
    mask = outliers == 1

    return X[mask], y[mask], iso_forest


def handle_imbalanced(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: Literal["SMOTE", "undersample", "combine"],
    random_state=42,
) -> (pd.DataFrame, pd.Series, object):
    """
    Balances an imbalanced dataset using techniques from imbalanced-learn.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        y: pd.Series
            Target array.
        strategy: Literal["SMOTE", "undersample", "combine"]
            Resampling method to use. Options:
            - 'SMOTE': Synthetic Minority Oversampling Technique.
            - 'undersample': Random undersampling of majority class.
            - 'combine': SMOTE with Tomek links.
        random_state: int
            Seed for reproducibility.

    Returns:
        X_resampled: pd.DataFrame
            Feature matrix after resampling.
        y_resampled: pd.Series
            Target array after resampling.
        transformer: object
            Fitted resampling method instance.
    """
    if strategy not in settings.resampler_mapping.keys():
        raise ValueError(
            f"Invalid resampling strategy, choose from {settings.resampler_mapping.keys()}."
        )

    y_series = y if isinstance(y, pd.Series) else pd.Series(y)
    minority_count = y_series.value_counts().min()
    if strategy in {"SMOTE", "combine"}:
        k_neighbors = min(5, minority_count - 1)
        if k_neighbors < 1:
            return X, y, None
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        if strategy == "SMOTE":
            resampler = smote
        else:
            resampler = SMOTETomek(random_state=random_state, smote=smote)
    else:
        resampler = settings.resampler_mapping[strategy](random_state=random_state)
    X_resampled, y_resampled = resampler.fit_resample(X, y)

    return X_resampled, y_resampled, resampler


def handle_skewed(
    X: pd.DataFrame, feature_names: List[str], threshold: float = 1
) -> (pd.DataFrame, PowerTransformer, List[str]):
    """
    Handles skewed features in the dataset using PowerTransformer.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        feature_names: List[str]
            Names of the features in the dataset.
        threshold: float
            Threshold for skewness.

    Returns:
        X_transformed: pd.DataFrame
            Feature matrix with skewed features transformed.
        transformer: PowerTransformer
            Fitted PowerTransformer model.
        skewed_feature_names: List[str]
            Names of the skewed features that were transformed.
        lambdas: List[float]
            Lambda values for the transformed features.
    """
    X = X.copy()
    skewed_feature_names = []
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    for feature in feature_names:
        feature_skewness = skew(X[feature])
        if abs(feature_skewness) > threshold:
            skewed_feature_names.append(feature)

    lambdas = []
    if len(skewed_feature_names) > 0:
        X[skewed_feature_names] = pt.fit_transform(X[skewed_feature_names])
        lambdas = pt.lambdas_

    return X, pt, skewed_feature_names, lambdas


def handle_missing_numeric(
    X: pd.DataFrame,
    feature_names: List[str],
    strategy: Literal["iterative", "knn", "mean", "median", "constant"],
) -> (pd.DataFrame, object):
    """
    Handles missing numeric values in the dataset using specified imputation strategy.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        feature_names: List[str]
            Names of the numeric features in the dataset.
        strategy: Literal["iterative", "knn", "mean", "median", "constant"]
            Imputation strategy to use.

    Returns:
        X_imputed: pd.DataFrame
            Feature matrix with missing numeric values imputed.
        imputer: object
            Fitted imputer model.
    """
    if strategy not in settings.imputer_mapping.keys():
        raise ValueError(
            f"Invalid imputation strategy, choose from {settings.imputer_mapping.keys()}."
        )

    X = X.copy()
    imputer = settings.imputer_mapping[strategy]()
    imputer.fit(X[feature_names])
    X[feature_names] = imputer.transform(X[feature_names])

    return X, imputer


def handle_missing_categorical(
    X: pd.DataFrame,
    feature_names: List[str],
    strategy: Literal["most_frequent", "constant"],
) -> (pd.DataFrame, SimpleImputer):
    """
    Handles missing categorical values in the dataset using specified imputation strategy.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        feature_names: List[str]
            Names of the categorical features in the dataset.
        strategy: Literal["most_frequent", "constant"]
            Imputation strategy to use.

    Returns:
        X_imputed: pd.DataFrame
            Feature matrix with missing categorical values imputed.
        imputer: SimpleImputer
            Fitted SimpleImputer model.
    """
    X = X.copy()
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X[feature_names])
    X[feature_names] = imputer.transform(X[feature_names])

    return X, imputer


def handle_categorical(
    X: pd.DataFrame, feature_names: List[str]
) -> (pd.DataFrame, OneHotEncoder):
    """
    Handles categorical features in the dataset using OneHotEncoder.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        feature_names: List[str]
            Names of the categorical features in the dataset.

    Returns:
        X_encoded: pd.DataFrame
            Feature matrix with categorical features encoded.
        encoder: OneHotEncoder
            Fitted OneHotEncoder model.
        ohe_feature_names: List[str]
            Names of the one-hot encoded features.
    """
    X = X.copy()
    encoder = OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False)
    encoder.fit(X[feature_names])
    encoded_features = encoder.transform(X[feature_names])
    ohe_feature_names = encoder.get_feature_names_out(feature_names)
    encoded_features_df = pd.DataFrame(
        encoded_features,
        columns=ohe_feature_names,
        index=X.index,
    )
    X = X.drop(columns=feature_names).join(encoded_features_df)

    return X, encoder, ohe_feature_names


def handle_scaling(
    X: pd.DataFrame, feature_names: List[str], strategy: Literal["standard", "robust"]
) -> (pd.DataFrame, object):
    """
    Handles scaling of features in the dataset using specified scaling strategy.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        feature_names: List[str]
            Names of the features to be scaled.
        strategy: Literal["standard", "robust"]
            Scaling strategy to use.

    Returns:
        X_scaled: pd.DataFrame
            Feature matrix with scaled features.
        scaler: object
            Fitted scaler instance.
    """
    if strategy not in ["standard", "robust"]:
        raise ValueError(
            f"Invalid scaling strategy, choose from {settings.scaler_mapping.keys()}."
        )

    X = X.copy()
    scaler = settings.scaler_mapping[strategy]()
    scaler.fit(X[feature_names])
    X[feature_names] = scaler.transform(X[feature_names])

    return X, scaler


def handle_selection(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.05, random_state: int = 42
) -> (pd.DataFrame, SelectFromModel, List[str]):
    """
    Handles feature selection using ExtraTreesClassifier.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        y: pd.Series
            Target array.
        threshold: float
            Threshold for feature selection.
        random_state: int
            Seed for reproducibility.

    Returns:
        X_selected: pd.DataFrame
            Feature matrix with selected features.
        selector: SelectFromModel
            Fitted SelectFromModel instance.
        selected_features: List[str]
            Names of the selected features.
        feature_importances: np.ndarray
            Feature importances from the ExtraTreesClassifier.
    """
    X = X.copy()
    selector = SelectFromModel(
        ExtraTreesClassifier(random_state=random_state), threshold=threshold
    )
    selector.fit(X, y)
    X_selected = selector.transform(X)
    selected_features = X.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    feature_importances = selector.estimator_.feature_importances_

    return X_selected_df, selector, selected_features, feature_importances


def handle_extraction(
    X: pd.DataFrame, threshold: float = 0.95, random_state: int = 42
) -> (np.ndarray, PCA):
    """
    Handles feature extraction using PCA.

    Parameters:
        X: pd.DataFrame
            Feature matrix.
        threshold: float
            Threshold for PCA.
        random_state: int
            Seed for reproducibility.

    Returns:
        X_extracted: np.ndarray
            Feature matrix after PCA transformation.
        extractor: PCA
            Fitted PCA instance.
        loadings: np.ndarray
            Loadings of the PCA components.
    """
    X = X.copy()
    extractor = PCA(
        n_components=threshold, svd_solver="full", random_state=random_state
    )
    extractor.fit(X)
    X_extracted = extractor.transform(X)
    loadings = extractor.components_

    return X_extracted, extractor, loadings
