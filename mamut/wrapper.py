import logging
import os
import time
import warnings
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from mamut.evidence import build_evidence_report
from mamut.preprocessing.preprocessing import Preprocessor
from mamut.utils.utils import metric_dict

from .evaluation import ModelEvaluator
from .model_selection import (
    CandidatePipelineClassifier,
    ModelSelector,
    available_model_names,
    fit_estimator,
    preprocessing_profile_for_model,
    repeated_cv_selection_decision,
)

log = logging.getLogger(__name__)


class LabelDecodedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap an encoded-label classifier so public predictions use original labels."""

    def __init__(self, estimator, label_encoder):
        self.estimator = estimator
        self.label_encoder = label_encoder

    def fit(self, X, y):
        y_encoded = self.label_encoder.transform(y)
        fit_estimator(self.estimator, X, y_encoded)
        return self

    def predict(self, X):
        encoded_predictions = self.estimator.predict(X)
        return self.label_encoder.inverse_transform(
            np.asarray(encoded_predictions).astype(int).ravel()
        )

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    @property
    def classes_(self):
        return self.label_encoder.classes_


class Mamut:
    """
    A class used to manage the machine learning pipeline, including preprocessing, model selection, and evaluation.

    Attributes
    ----------
    preprocess : bool
        Whether to apply preprocessing to the data.
    imb_threshold : float
        Threshold for detecting imbalanced data.
    exclude_models : Optional[List[str]]
        List of models to exclude from selection.
    score_metric : callable
        Metric used to evaluate model performance.
    optimization_method : Literal["random_search", "bayes"]
        Method for hyperparameter optimization.
    n_iterations : Optional[int]
        Number of iterations for optimization.
    random_state : Optional[int]
        Random state for reproducibility.
    preprocessor : Preprocessor
        Preprocessor object for data preprocessing.
    le : LabelEncoder
        Label encoder for target variable.
    model_selector : ModelSelector
        Object for model selection.
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target variable.
    X_train : pd.DataFrame
        Training features.
    X_validation : pd.DataFrame
        Validation features used for model selection.
    y_train : pd.Series
        Training target variable.
    y_validation : pd.Series
        Validation target variable used for model selection.
    X_holdout : Optional[pd.DataFrame]
        Optional final holdout features used only for final evaluation.
    y_holdout : Optional[pd.Series]
        Optional final holdout target used only for final evaluation.
    raw_fitted_models_ : Optional[List[Pipeline]]
        List of raw fitted models.
    fitted_models_ : Optional[List[Pipeline]]
        List of fitted models with preprocessing.
    best_model_ : Optional[Pipeline]
        Best model pipeline.
    best_score_ : float
        Best model score.
    training_summary_ : dict
        Summary of the training process.
    optuna_studies_ : dict
        Optuna studies for hyperparameter optimization.
    ensemble_ : Optional[Pipeline]
        Ensemble model pipeline.
    greedy_ensemble_ : Optional[Pipeline]
        Greedy ensemble model pipeline.
    ensemble_models_ : Optional[List[Pipeline]]
        List of models in the ensemble.
    imbalanced_ : bool
        Whether the data is imbalanced.

    Methods
    -------
    fit(X: pd.DataFrame, y: pd.Series) -> Pipeline
        Fits the model to the data.
    predict(X: pd.DataFrame) -> np.ndarray
        Predicts the target variable for the given data.
    predict_proba(X: pd.DataFrame) -> np.ndarray
        Predicts the probabilities of the target variable for the given data.
    evaluate() -> None
        Evaluates the fitted models.
    save_best_model(path: str) -> None
        Saves the best model to the specified path.
    create_ensemble(voting: Literal["soft", "hard"] = "soft") -> Pipeline
        Creates an ensemble of the fitted models.
    create_greedy_ensemble(n_models: int = 6, voting: Literal["soft", "hard"] = "soft") -> Pipeline
        Creates a greedy ensemble of the fitted models.
    """

    def __init__(
        self,
        preprocess: bool = True,
        imb_threshold: float = 0.10,
        exclude_models: Optional[List[str]] = None,
        include_models: Optional[List[str]] = None,
        score_metric: Literal[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "jaccard",
            "roc_auc_score",
        ] = "f1",
        search_profile: Literal["quick", "balanced", "thorough"] = "balanced",
        optimization_method: Literal["random_search", "bayes"] = "bayes",
        n_iterations: int = 30,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = 1,
        selection_strategy: Literal[
            "single_split", "nested_cv", "repeated_cv"
        ] = "single_split",
        selection_cv_splits: int = 5,
        selection_cv_repeats: int = 2,
        selection_practical_margin: float = 0.005,
        preprocessing_profile: Literal["auto", "generic_ohe"] = "auto",
        validation_size: float = 0.2,
        holdout_size: Optional[float] = None,
        save_models: bool = False,
        models_output_dir: str = "fitted_models",
        refit_final_model: bool = False,
        verbose: bool = False,
        evidence_cv_splits: int = 5,
        evidence_cv_repeats: int = 3,
        evidence_confidence_level: float = 0.95,
        evidence_practical_margin: float = 0.01,
        **preprocessor_kwargs,
    ):
        """
        Constructs all the necessary attributes for the Mamut object.

        Parameters
        ----------
        preprocess : bool
            Whether to apply preprocessing to the data.
        imb_threshold : float
            Threshold for detecting imbalanced data.
        exclude_models : Optional[List[str]]
            List of models to exclude from selection.
        include_models : Optional[List[str]]
            Optional exact list of models to include. Mutually exclusive with
            exclude_models.
        score_metric : Literal["accuracy", "precision", "recall", "f1", "balanced_accuracy", "jaccard", "roc_auc_score"]
            Metric used to evaluate model performance.
        search_profile : Literal["quick", "balanced", "thorough"]
            Candidate set to search when include_models is not supplied.
        optimization_method : Literal["random_search", "bayes"]
            Method for hyperparameter optimization.
        n_iterations : Optional[int]
            Number of iterations for optimization.
        random_state : Optional[int]
            Random state for reproducibility.
        n_jobs : Optional[int]
            Number of worker threads for supported estimators. Use None to
            keep estimator defaults.
        selection_strategy : Literal["single_split", "nested_cv", "repeated_cv"]
            Whether to select the final candidate by one validation split or
            by nested CV over non-holdout modeling data. ``repeated_cv`` is a
            deprecated alias for ``nested_cv``.
        selection_cv_splits : int
            Number of stratified folds for repeated-CV selection.
        selection_cv_repeats : int
            Number of repeats for repeated-CV selection.
        selection_practical_margin : float
            Maximum mean-score difference treated as a practical tie before
            preferring lower score variance and faster runtime.
        preprocessing_profile : Literal["auto", "generic_ohe"]
            ``auto`` lets each candidate use a model-aware preprocessing
            profile. ``generic_ohe`` preserves the legacy shared one-hot
            preprocessing behavior.
        validation_size : float
            Fraction of the modeling data reserved for model selection.
        holdout_size : Optional[float]
            Optional fraction of the original data reserved for final evaluation.
            Holdout data is never used for model or ensemble selection.
        save_models : bool
            Whether to save fitted candidate models during fit.
        models_output_dir : str
            Directory for fitted model artifacts when save_models=True.
        evidence_cv_splits : int
            Number of stratified folds used in evidence score stability checks.
        evidence_cv_repeats : int
            Number of repeats used in evidence score stability checks.
        evidence_confidence_level : float
            Confidence level used for evidence score intervals.
        evidence_practical_margin : float
            Minimum metric difference required before evidence challenges the
            validation-selected model.
        **preprocessor_kwargs
            Additional keyword arguments for the Preprocessor.
        """
        if score_metric not in metric_dict:
            valid_metrics = ", ".join(sorted(metric_dict))
            raise ValueError(f"score_metric must be one of: {valid_metrics}.")
        if optimization_method not in {"random_search", "bayes"}:
            raise ValueError(
                "optimization_method must be one of: 'random_search', 'bayes'."
            )
        if search_profile not in {"quick", "balanced", "thorough"}:
            raise ValueError(
                "search_profile must be one of: 'quick', 'balanced', 'thorough'."
            )
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError(
                "n_iterations must be an integer greater than or equal to 1."
            )
        if n_jobs is not None and (
            not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1
        ):
            raise ValueError("n_jobs must be None, -1, or a positive integer.")
        if selection_strategy not in {"single_split", "nested_cv", "repeated_cv"}:
            raise ValueError(
                "selection_strategy must be one of: 'single_split', 'nested_cv', "
                "'repeated_cv'."
            )
        if selection_cv_splits < 2:
            raise ValueError("selection_cv_splits must be at least 2.")
        if selection_cv_repeats < 1:
            raise ValueError("selection_cv_repeats must be at least 1.")
        if selection_practical_margin < 0:
            raise ValueError("selection_practical_margin must be non-negative.")
        if preprocessing_profile not in {"auto", "generic_ohe"}:
            raise ValueError(
                "preprocessing_profile must be one of: 'auto', 'generic_ohe'."
            )
        self._validate_split_size(validation_size, "validation_size")
        if holdout_size is not None:
            self._validate_split_size(holdout_size, "holdout_size")
        if evidence_cv_splits < 2:
            raise ValueError("evidence_cv_splits must be at least 2.")
        if evidence_cv_repeats < 1:
            raise ValueError("evidence_cv_repeats must be at least 1.")
        if not 0 < evidence_confidence_level < 1:
            raise ValueError(
                "evidence_confidence_level must be greater than 0 and less than 1."
            )
        if evidence_practical_margin < 0:
            raise ValueError("evidence_practical_margin must be non-negative.")
        if include_models and exclude_models:
            raise ValueError("Use include_models or exclude_models, not both.")
        known_models = set(available_model_names())
        exclude_models = list(exclude_models or [])
        include_models = list(include_models) if include_models is not None else None
        unknown_models = sorted(set(exclude_models) - known_models)
        if unknown_models:
            valid_models = ", ".join(available_model_names())
            raise ValueError(
                f"exclude_models contains unsupported model names: {unknown_models}. "
                f"Valid model names are: {valid_models}."
            )
        if include_models is not None:
            unknown_models = sorted(set(include_models) - known_models)
            if unknown_models:
                valid_models = ", ".join(available_model_names())
                raise ValueError(
                    f"include_models contains unsupported model names: {unknown_models}. "
                    f"Valid model names are: {valid_models}."
                )

        self.preprocess = preprocess
        self.imb_threshold = imb_threshold
        self.exclude_models = exclude_models
        self.include_models = include_models
        self.score_metric = metric_dict[score_metric]
        self.score_metric_name = self.score_metric.__name__
        self.search_profile = search_profile
        self.optimization_method = optimization_method
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.selection_strategy = selection_strategy
        self.selection_cv_splits = selection_cv_splits
        self.selection_cv_repeats = selection_cv_repeats
        self.selection_practical_margin = selection_practical_margin
        self.preprocessing_profile = preprocessing_profile
        self.validation_size = validation_size
        self.holdout_size = holdout_size
        self.save_models = save_models
        self.models_output_dir = models_output_dir
        self.refit_final_model = refit_final_model
        self.verbose = verbose
        self.evidence_cv_splits = evidence_cv_splits
        self.evidence_cv_repeats = evidence_cv_repeats
        self.evidence_confidence_level = evidence_confidence_level
        self.evidence_practical_margin = evidence_practical_margin
        self.preprocessor_kwargs = preprocessor_kwargs.copy()
        self.preprocessor_kwargs.setdefault("imbalance_threshold", imb_threshold)
        self.imb_threshold = self.preprocessor_kwargs["imbalance_threshold"]

        self.preprocessor = (
            Preprocessor(**self.preprocessor_kwargs) if preprocess else None
        )
        self.le = LabelEncoder()
        self.model_selector = None

        self.X = None
        self.y = None
        self.y_encoded_ = None
        self.X_train = None
        self.X_validation = None
        self.X_holdout = None
        self.y_train = None
        self.y_validation = None
        self.y_holdout = None
        self.X_modeling_raw_ = None
        self.y_modeling_raw_ = None
        self.y_modeling_original_ = None
        self.X_train_raw_ = None
        self.X_validation_raw_ = None
        self.y_train_raw_ = None
        self.y_validation_raw_ = None
        self.X_holdout_raw_ = None
        self.y_holdout_original_ = None
        self.groups_ = None
        self.groups_modeling_ = None
        self.groups_train_ = None
        self.groups_validation_ = None
        self.groups_holdout_ = None
        self.X_test = None
        self.y_test = None
        self.binary = None
        self.roc = None

        self.raw_fitted_models_ = None
        self.candidate_pipelines_ = None
        self.fitted_preprocessors_ = None
        self.fitted_training_data_ = None
        self.fitted_validation_data_ = None
        self.fitted_models_ = None
        self.selected_estimator_ = None
        self.validation_selected_estimator_ = None
        self.final_preprocessor_ = None
        self.final_estimator_ = None
        self.best_model_ = None

        self.best_score_ = None
        self.best_validation_score_ = None
        self.holdout_score_ = None
        self.validation_summary_ = None
        self.holdout_summary_ = None
        self.training_summary_ = None
        self.selection_summary_ = None
        self.optuna_studies_ = None
        self.models_output_path_ = None
        self.evidence_report_ = None
        self.validation_integrity_ = None
        self.leakage_checks_ = None
        self.baseline_comparison_ = None
        self.score_stability_ = None
        self.selection_guidance_ = None
        self.selection_summary_ = None
        self.report_result_ = None

        self.ensemble_ = None
        self.greedy_ensemble_ = None
        self.ensemble_models_ = None
        self.imbalanced_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_holdout: Optional[pd.DataFrame] = None,
        y_holdout: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
        groups_holdout: Optional[pd.Series] = None,
    ):
        """
        Fits the model to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series
            The target variable.
        X_holdout : Optional[pd.DataFrame]
            Optional final holdout features. If provided, y_holdout must also
            be provided and holdout_size must be None.
        y_holdout : Optional[pd.Series]
            Optional final holdout target. Holdout rows are never used for
            model or ensemble selection.
        groups : Optional[pd.Series]
            Group labels for observations that must remain in the same fold.
        groups_holdout : Optional[pd.Series]
            Group labels for explicit holdout rows. Required with ``groups``
            and explicit holdout data so overlap can be rejected.

        Returns
        -------
        Pipeline
            The best model pipeline.
        """
        if (X_holdout is None) != (y_holdout is None):
            raise ValueError("X_holdout and y_holdout must be provided together.")
        if X_holdout is not None and self.holdout_size is not None:
            raise ValueError(
                "Use either holdout_size or explicit X_holdout/y_holdout, not both."
            )
        if groups_holdout is not None and X_holdout is None:
            raise ValueError("groups_holdout requires explicit X_holdout/y_holdout.")
        if groups_holdout is not None and groups is None:
            raise ValueError("groups_holdout requires groups.")
        if X_holdout is not None and groups is not None and groups_holdout is None:
            raise ValueError(
                "groups_holdout is required with groups and explicit holdout data."
            )

        self.preprocessor = (
            Preprocessor(**self.preprocessor_kwargs) if self.preprocess else None
        )
        self.X_holdout = None
        self.y_holdout = None
        self.X_holdout_raw_ = None
        self.y_holdout_original_ = None
        self.holdout_summary_ = None
        self.holdout_score_ = None
        self.models_output_path_ = None
        self.evidence_report_ = None
        self.validation_integrity_ = None
        self.leakage_checks_ = None
        self.baseline_comparison_ = None
        self.score_stability_ = None
        self.selection_guidance_ = None
        self.report_result_ = None
        self.selected_estimator_ = None
        self.validation_selected_estimator_ = None
        self.final_preprocessor_ = None
        self.final_estimator_ = None
        self.fitted_preprocessors_ = None
        self.fitted_training_data_ = None
        self.fitted_validation_data_ = None
        self.candidate_pipelines_ = None
        self.groups_ = None
        self.groups_modeling_ = None
        self.groups_train_ = None
        self.groups_validation_ = None
        self.groups_holdout_ = None
        self.imbalanced_ = False

        Mamut._check_categorical(y)
        y_original = pd.Series(y).copy()
        y_original.index = X.index
        if y_original.value_counts(normalize=True).min() < self.imb_threshold:
            self.imbalanced_ = True

        y_encoded = pd.Series(
            self.le.fit_transform(y_original),
            index=X.index,
            name=y_original.name,
        )
        groups_encoded = None
        if groups is not None:
            if len(groups) != len(X):
                raise ValueError("groups must have the same length as X.")
            groups_encoded = pd.Series(groups).copy()
            groups_encoded.index = X.index
        self.groups_ = groups_encoded

        X_modeling = X.copy()
        y_modeling = y_encoded.copy()
        y_modeling_original = y_original.copy()
        groups_modeling = groups_encoded.copy() if groups_encoded is not None else None

        if X_holdout is not None:
            Mamut._check_categorical(pd.Series(y_holdout))
            y_holdout_original = pd.Series(y_holdout).copy()
            y_holdout_original.index = X_holdout.index
            y_holdout_encoded = pd.Series(
                self.le.transform(y_holdout_original),
                index=X_holdout.index,
                name=y_holdout_original.name,
            )
            self.X_holdout_raw_ = X_holdout.copy()
            self.y_holdout_original_ = y_holdout_original
            if groups_holdout is not None:
                if len(groups_holdout) != len(X_holdout):
                    raise ValueError(
                        "groups_holdout must have the same length as X_holdout."
                    )
                self.groups_holdout_ = pd.Series(groups_holdout).copy()
                self.groups_holdout_.index = X_holdout.index
                overlap = set(groups_modeling).intersection(self.groups_holdout_)
                if overlap:
                    raise ValueError(
                        "Explicit holdout groups overlap modeling groups; "
                        "holdout evaluation would not be independent."
                    )
        elif self.holdout_size is not None:
            (
                X_modeling,
                self.X_holdout_raw_,
                y_modeling,
                y_holdout_encoded,
                y_modeling_original,
                self.y_holdout_original_,
                groups_modeling,
                self.groups_holdout_,
            ) = self._split_data(
                X_modeling,
                y_modeling,
                y_modeling_original,
                groups_modeling,
                test_size=self.holdout_size,
            )
        else:
            y_holdout_encoded = None

        self.X_modeling_raw_ = X_modeling.copy()
        self.y_modeling_raw_ = y_modeling.copy()
        self.y_modeling_original_ = y_modeling_original.copy()
        self.groups_modeling_ = (
            groups_modeling.copy() if groups_modeling is not None else None
        )

        (
            X_train_raw,
            X_validation_raw,
            y_train,
            y_validation,
            _,
            _,
            self.groups_train_,
            self.groups_validation_,
        ) = self._split_data(
            X_modeling,
            y_modeling,
            y_modeling_original,
            groups_modeling,
            test_size=self.validation_size,
        )

        X_train = X_train_raw
        X_validation = X_validation_raw
        self.y_train_raw_ = y_train.copy()
        self.y_validation_raw_ = y_validation.copy()
        use_shared_preprocessor = self.preprocess and (
            self.preprocessing_profile == "generic_ohe"
        )
        if use_shared_preprocessor:
            X_train, y_train = self.preprocessor.fit_transform(X_train, y_train)
            X_validation = self.preprocessor.transform(X_validation)
            if self.X_holdout_raw_ is not None:
                self.X_holdout = self.preprocessor.transform(self.X_holdout_raw_)
        elif self.X_holdout_raw_ is not None:
            self.X_holdout = self.X_holdout_raw_

        self.X_train = self._as_model_input(X_train)
        self.X_validation = self._as_model_input(X_validation)
        self.y_train = np.asarray(y_train)
        self.y_validation = np.asarray(y_validation)
        if self.X_holdout is not None:
            self.X_holdout = self._as_model_input(self.X_holdout)
            self.y_holdout = np.asarray(y_holdout_encoded)

        self.X_train_raw_ = X_train_raw
        self.X_validation_raw_ = X_validation_raw
        self.X = X.copy()
        self.y = y_original
        self.y_encoded_ = y_encoded

        # Backward-compatible aliases. These represent validation data, not a
        # final test set.
        self.X_test = self.X_validation
        self.y_test = self.y_validation

        self.model_selector = ModelSelector(
            self.X_train,
            self.y_train,
            self.X_validation,
            self.y_validation,
            X_train_raw=self.X_train_raw_,
            y_train_raw=self.y_train_raw_,
            X_validation_raw=self.X_validation_raw_,
            y_validation_raw=self.y_validation_raw_,
            groups_train_raw=self.groups_train_,
            preprocessor_factory=self._make_model_preprocessor,
            exclude_models=self.exclude_models,
            include_models=self.include_models,
            search_profile=self.search_profile,
            score_metric=self.score_metric,
            optimization_method=self.optimization_method,
            n_iterations=self.n_iterations,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        (
            best_model,
            _,
            score_for_best_model,
            fitted_models,
            fitted_preprocessors,
            fitted_training_data,
            fitted_validation_data,
            validation_summary,
            studies,
        ) = self.model_selector.compare_models()

        self.raw_fitted_models_ = fitted_models
        self.fitted_preprocessors_ = fitted_preprocessors
        self.fitted_training_data_ = fitted_training_data
        self.fitted_validation_data_ = fitted_validation_data
        self.optuna_studies_ = studies
        self.fitted_models_ = [
            self._make_public_pipeline(model, self.fitted_preprocessors_[model_name])
            for model_name, model in fitted_models.items()
        ]

        # Update the score metric based on binary/multiclass problem (for ensembles)
        self.score_metric = self.model_selector.score_metric
        self.score_metric_name = self.model_selector.score_metric_name
        self.binary = self.model_selector.binary
        self.roc = self.model_selector.roc
        validation_summary = validation_summary.sort_values(
            by=self.score_metric_name, ascending=False
        ).reset_index(drop=True)
        self.best_validation_score_ = score_for_best_model
        self.validation_selected_estimator_ = best_model
        self.selected_estimator_ = best_model
        self.selection_summary_ = self._build_selection_summary(
            validation_summary, selected_model_name=best_model.__class__.__name__
        )
        if self.selection_strategy in {"nested_cv", "repeated_cv"}:
            if self.selection_strategy == "repeated_cv":
                warnings.warn(
                    "selection_strategy='repeated_cv' is deprecated; use "
                    "'nested_cv' for nested, fold-local model selection.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.selection_summary_ = self.model_selector.nested_cv_selection_scores(
                X=self.X_modeling_raw_,
                y=self.y_modeling_raw_,
                groups=self.groups_modeling_,
                cv_splits=self.selection_cv_splits,
                cv_repeats=self.selection_cv_repeats,
                confidence_level=self.evidence_confidence_level,
            )
            selection_decision = repeated_cv_selection_decision(
                self.selection_summary_,
                practical_margin=self.selection_practical_margin,
            )
            selected_model_name = selection_decision["model"]
            self.selected_estimator_ = self.raw_fitted_models_[selected_model_name]
            self.selection_summary_ = self._mark_selected_model(
                self.selection_summary_,
                selected_model_name=selected_model_name,
                selection_strategy="nested_cv",
                selection_decision_status=selection_decision["status"],
            )
            score_for_best_model = float(
                self.selection_summary_.loc[
                    self.selection_summary_["model"].eq(selected_model_name),
                    "mean_score",
                ].iloc[0]
            )
        self.best_score_ = score_for_best_model
        selected_model_name = self.selected_estimator_.__class__.__name__
        public_preprocessor = self.fitted_preprocessors_.get(selected_model_name)
        self.preprocessor = public_preprocessor
        self.X_train, self.y_train = self.fitted_training_data_[selected_model_name]
        self.X_validation, self.y_validation = self.fitted_validation_data_[
            selected_model_name
        ]
        if self.X_holdout_raw_ is not None:
            self.X_holdout = self._transform_for_model(
                selected_model_name, self.X_holdout_raw_
            )
        if self.refit_final_model:
            if self.selection_strategy in {"nested_cv", "repeated_cv"}:
                self.final_preprocessor_, self.final_estimator_ = (
                    self._retune_selected_model_on_modeling_data(selected_model_name)
                )
            else:
                self.final_preprocessor_, self.final_estimator_ = (
                    self._refit_selected_model(self.selected_estimator_)
                )
            self.selected_estimator_ = self.final_estimator_
            public_preprocessor = self.final_preprocessor_
            self.preprocessor = public_preprocessor
            if self.X_holdout_raw_ is not None:
                if self.final_preprocessor_ is not None:
                    self.X_holdout = self.final_preprocessor_.transform(
                        self.X_holdout_raw_.copy()
                    )
                else:
                    self.X_holdout = self._as_model_input(self.X_holdout_raw_)

        self.best_model_ = self._make_public_pipeline(
            self.selected_estimator_, public_preprocessor
        )
        self.validation_summary_ = validation_summary
        self.training_summary_ = validation_summary
        self.holdout_summary_ = (
            self._score_models_on_dataset(
                self.raw_fitted_models_, self.X_holdout_raw_, self.y_holdout
            )
            if self.X_holdout_raw_ is not None
            else None
        )
        if self.holdout_summary_ is not None and self.refit_final_model:
            final_scores = self._score_model_with_metrics(
                self.selected_estimator_,
                self.X_holdout,
                self.y_holdout,
            )
            selected_row = self.holdout_summary_["model"].eq(selected_model_name)
            for metric_name, score in final_scores.items():
                self.holdout_summary_.loc[selected_row, metric_name] = score
        self.holdout_score_ = (
            self._score_selected_model_on_holdout()
            if self.X_holdout_raw_ is not None
            else None
        )

        log.info(f"Best model: {best_model.__class__.__name__}")

        if self.save_models:
            self._save_fitted_models()

        return self.best_model_

    def _build_selection_summary(
        self, validation_summary: pd.DataFrame, *, selected_model_name: str
    ) -> pd.DataFrame:
        summary = validation_summary.copy()
        summary["metric"] = self.score_metric_name
        summary["mean_score"] = summary[self.score_metric_name]
        summary["std_score"] = np.nan
        summary["ci_low"] = np.nan
        summary["ci_high"] = np.nan
        summary["n_scores"] = 1
        summary["selection_duration"] = summary["duration"]
        summary["status"] = "ok"
        return self._mark_selected_model(
            summary,
            selected_model_name=selected_model_name,
            selection_strategy="single_split",
        )

    @staticmethod
    def _mark_selected_model(
        summary: pd.DataFrame,
        *,
        selected_model_name: str,
        selection_strategy: str,
        selection_decision_status: str = "confirmed",
    ) -> pd.DataFrame:
        summary = summary.copy()
        summary["selected"] = summary["model"].eq(selected_model_name)
        summary["selection_strategy"] = selection_strategy
        summary["selection_decision_status"] = selection_decision_status
        return summary

    def _save_fitted_models(self) -> None:
        models_dir = os.path.join(
            os.getcwd(),
            self.models_output_dir,
            str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())),
        )
        os.makedirs(models_dir, exist_ok=True)
        self.models_output_path_ = models_dir

        for model in self.fitted_models_:
            model_name = self._pipeline_model_name(model)
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            log.info(f"Saved model {model_name} to {model_path}")

    def _make_public_pipeline(self, estimator, preprocessor) -> Pipeline:
        steps = []
        if preprocessor is not None:
            steps.append(("preprocessor", preprocessor))
        steps.append(("model", LabelDecodedClassifier(estimator, self.le)))
        return Pipeline(steps)

    @staticmethod
    def _pipeline_model_name(model: Pipeline) -> str:
        final_step = model.named_steps["model"]
        estimator = getattr(final_step, "estimator", final_step)
        return estimator.__class__.__name__

    def _refit_selected_model(self, selected_estimator):
        selected_model_name = selected_estimator.__class__.__name__
        preprocessor = self._make_model_preprocessor(selected_model_name)
        if preprocessor is not None:
            X_final, y_final = preprocessor.fit_transform(
                self.X_modeling_raw_.copy(),
                self.y_modeling_raw_.copy(),
            )
        else:
            X_final = self._as_model_input(self.X_modeling_raw_)
            y_final = np.asarray(self.y_modeling_raw_)

        final_estimator = clone(selected_estimator)
        fit_estimator(final_estimator, X_final, y_final)
        return preprocessor, final_estimator

    def _retune_selected_model_on_modeling_data(self, selected_model_name: str):
        final_selector = ModelSelector(
            self._as_model_input(self.X_modeling_raw_),
            np.asarray(self.y_modeling_raw_),
            self._as_model_input(self.X_validation_raw_),
            np.asarray(self.y_validation_raw_),
            score_metric=self.model_selector.base_score_metric,
            X_train_raw=self.X_modeling_raw_,
            y_train_raw=self.y_modeling_raw_,
            X_validation_raw=self.X_validation_raw_,
            y_validation_raw=self.y_validation_raw_,
            groups_train_raw=self.groups_modeling_,
            preprocessor_factory=self._make_model_preprocessor,
            include_models=[selected_model_name],
            search_profile=self.search_profile,
            optimization_method=self.optimization_method,
            n_iterations=self.n_iterations,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        estimator, preprocessor, _, _, study = final_selector.tune_and_fit_model(
            selected_model_name
        )
        self.optuna_studies_[selected_model_name] = study
        return preprocessor, estimator

    def _make_candidate_pipeline(self, model_name: str, estimator):
        preprocessor = self._make_model_preprocessor(model_name)
        profile = preprocessor.profile if preprocessor is not None else "generic_ohe"
        return CandidatePipelineClassifier(
            estimator=clone(estimator),
            preprocess=preprocessor is not None,
            profile=profile,
            preprocessor_kwargs=self.preprocessor_kwargs.copy(),
        )

    def _ensemble_estimators(self, model_names=None):
        names = model_names or list(self.raw_fitted_models_)
        return [
            (name, self._make_candidate_pipeline(name, self.raw_fitted_models_[name]))
            for name in names
        ]

    def _fitted_candidate_pipelines(self) -> dict:
        pipelines = {}
        selected_name = (
            self.selected_estimator_.__class__.__name__
            if self.selected_estimator_ is not None
            else None
        )
        for model_name, estimator in self.raw_fitted_models_.items():
            if self.refit_final_model and model_name == selected_name:
                estimator = self.selected_estimator_
                preprocessor = self.final_preprocessor_
            else:
                preprocessor = self.fitted_preprocessors_.get(model_name)
            pipeline = self._make_candidate_pipeline(model_name, estimator)
            pipeline.estimator_ = estimator
            pipeline.preprocessor_ = preprocessor
            pipeline.classes_ = estimator.classes_
            pipelines[model_name] = pipeline
        return pipelines

    def _split_data(self, X, y, y_original, groups, *, test_size: float):
        if groups is None:
            X_train, X_eval, y_train, y_eval, y_original_train, y_original_eval = (
                train_test_split(
                    X,
                    y,
                    y_original,
                    test_size=test_size,
                    stratify=y,
                    random_state=self.random_state,
                )
            )
            return (
                X_train,
                X_eval,
                y_train,
                y_eval,
                y_original_train,
                y_original_eval,
                None,
                None,
            )

        target_distribution = pd.Series(y).value_counts(normalize=True)
        splitter = GroupShuffleSplit(
            n_splits=32,
            test_size=test_size,
            random_state=self.random_state,
        )
        candidates = []
        for train_idx, eval_idx in splitter.split(X, y, groups):
            y_train = y.iloc[train_idx]
            y_eval = y.iloc[eval_idx]
            if any(
                (
                    y_train.nunique() < target_distribution.size,
                    y_eval.nunique() < target_distribution.size,
                )
            ):
                continue
            eval_distribution = y_eval.value_counts(normalize=True).reindex(
                target_distribution.index, fill_value=0.0
            )
            balance_error = float((eval_distribution - target_distribution).abs().sum())
            size_error = abs((len(eval_idx) / len(X)) - test_size)
            candidates.append((balance_error + size_error, train_idx, eval_idx))

        if not candidates:
            raise ValueError(
                "Unable to construct a group-disjoint split containing every "
                "target class in both partitions."
            )
        _, train_idx, eval_idx = min(candidates, key=lambda candidate: candidate[0])
        return (
            X.iloc[train_idx],
            X.iloc[eval_idx],
            y.iloc[train_idx],
            y.iloc[eval_idx],
            y_original.iloc[train_idx],
            y_original.iloc[eval_idx],
            groups.iloc[train_idx],
            groups.iloc[eval_idx],
        )

    def _score_selected_model_on_holdout(self) -> float:
        if self.refit_final_model:
            if self.final_preprocessor_ is not None:
                X_holdout = self.final_preprocessor_.transform(self.X_holdout_raw_)
            else:
                X_holdout = self._as_model_input(self.X_holdout_raw_)
            return self._score_model_on_dataset(
                self.selected_estimator_, X_holdout, self.y_holdout
            )

        return self._score_model_on_dataset(
            self.selected_estimator_,
            self._transform_for_model(
                self.selected_estimator_.__class__.__name__, self.X_holdout_raw_
            ),
            self.y_holdout,
        )

    @staticmethod
    def _as_model_input(data):
        if isinstance(data, pd.DataFrame):
            return data.values
        return data

    def _score_models_on_dataset(self, models: dict, X, y) -> pd.DataFrame:
        duration_by_model = {}
        if isinstance(self.validation_summary_, pd.DataFrame):
            duration_by_model = dict(
                zip(
                    self.validation_summary_["model"],
                    self.validation_summary_["duration"],
                )
            )

        model_order = (
            self.validation_summary_["model"].to_list()
            if isinstance(self.validation_summary_, pd.DataFrame)
            else list(models)
        )
        rows = []
        for model_name in model_order:
            model = models[model_name]
            X_model = self._transform_for_model(model_name, X)
            rows.append(
                {
                    "model": model_name,
                    **self._score_model_with_metrics(model, X_model, y),
                    "duration": duration_by_model.get(model_name, np.nan),
                }
            )

        return pd.DataFrame(rows)

    def _transform_for_model(self, model_name: str, X: pd.DataFrame):
        if X is None:
            return None
        preprocessor = (
            self.fitted_preprocessors_.get(model_name)
            if isinstance(self.fitted_preprocessors_, dict)
            else None
        )
        if preprocessor is None:
            return self._as_model_input(X)
        return preprocessor.transform(X.copy())

    def _score_model_with_metrics(self, fitted_model, X, y) -> dict:
        y = np.asarray(y)
        y_pred = fitted_model.predict(X)
        y_pred_proba = fitted_model.predict_proba(X)
        if self.binary:
            y_pred_proba = y_pred_proba[:, 1]

        try:
            roc_auc = roc_auc_score(
                y,
                y_pred_proba,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            roc_auc = np.nan

        results = {
            "accuracy_score": accuracy_score(y, y_pred),
            "balanced_accuracy_score": balanced_accuracy_score(y, y_pred),
            "precision_score": precision_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "recall_score": recall_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "f1_score": f1_score(y, y_pred, average="weighted", zero_division=0),
            "jaccard_score": jaccard_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "roc_auc_score": roc_auc,
        }

        return {
            self.score_metric_name: results.pop(self.score_metric_name),
            **results,
        }

    def _score_model_on_dataset(self, fitted_model, X, y) -> float:
        if self.roc:
            if self.binary:
                predictions = fitted_model.predict_proba(X)[:, 1]
            else:
                predictions = fitted_model.predict_proba(X)
        else:
            predictions = fitted_model.predict(X)

        return self.score_metric(y, predictions)

    def predict(self, X: pd.DataFrame):
        """
        Predicts the target variable for the given data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.

        Returns
        -------
        np.ndarray
            Predicted target variable.
        """
        return self._predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """
        Predicts the probabilities of the target variable for the given data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.

        Returns
        -------
        np.ndarray
            Predicted probabilities of the target variable.
        """
        return self._predict(X, proba=True)

    def evaluate(
        self,
        n_top_models: int = 3,
        dataset: Literal["auto", "validation", "holdout"] = "auto",
        include_evidence: bool = True,
        output_dir: str = "mamut_report",
        include_shap: bool = True,
        shap_max_samples: Optional[int] = 200,
        display_plots: bool = False,
        write_html: bool = True,
        save_plots: bool = True,
    ) -> dict:
        """
        Evaluates the fitted models.
        """
        self._check_fitted()
        if not isinstance(n_top_models, int) or n_top_models < 1:
            raise ValueError(
                "n_top_models must be an integer greater than or equal to 1."
            )
        _, y_evaluation, evaluation_summary, evaluation_dataset = (
            self._get_evaluation_dataset(dataset)
        )
        evidence_report = (
            self.generate_evidence(dataset=evaluation_dataset)
            if include_evidence
            else None
        )

        evaluator = ModelEvaluator(
            self._fitted_candidate_pipelines(),
            X_evaluation=(
                self.X_holdout_raw_
                if evaluation_dataset == "holdout"
                else self.X_validation_raw_
            ),
            y_evaluation=y_evaluation,
            X_train=self.X_train,
            y_train=self.y_train,
            X_explanation=self.X_train_raw_,
            X=self.X,
            y=self.y,
            optimizer=self.optimization_method,
            metric=self.score_metric_name,
            n_trials=self.n_iterations,
            excluded_models=self.exclude_models,
            studies=self.optuna_studies_,
            training_summary=evaluation_summary,
            pca_loadings=(
                self.preprocessor.pca_loadings_ if self.preprocessor else None
            ),
            binary=self.model_selector.binary,
            preprocessing_steps=self.preprocessor.report() if self.preprocessor else {},
            feature_names=(
                self.preprocessor.feature_names_out_
                if self.preprocessor and self.preprocessor.feature_names_out_
                else self.X.columns.tolist()
            ),
            n_top_models=n_top_models,
            is_ensemble=self.greedy_ensemble_ is not None,
            greedy_ensemble=self.greedy_ensemble_,
            evaluation_dataset=evaluation_dataset,
            selected_model_name=self.selected_estimator_.__class__.__name__,
            rank_by_metric=evaluation_dataset == "validation",
            evidence_report=evidence_report,
            report_output_path=output_dir,
            include_shap=include_shap,
            shap_max_samples=shap_max_samples,
            write_html=write_html,
            save_plots=save_plots,
        )

        evaluator.evaluate_to_html(evaluation_summary)
        if display_plots:
            evaluator.plot_results_in_notebook()
        self.report_result_ = getattr(evaluator, "report_result_", None)
        return self.report_result_

    def generate_evidence(
        self,
        dataset: Literal["auto", "validation", "holdout"] = "auto",
        include_candidate_comparison: bool = True,
    ) -> dict:
        """Build diagnostic evidence without changing the fitted candidate.

        Parameters
        ----------
        dataset : Literal["auto", "validation", "holdout"]
            Evaluation partition to summarize.
        include_candidate_comparison : bool
            Whether to score non-selected MAMUT candidates alongside fixed
            baselines. Disable this for a locked final confirmation analysis.
        """
        self._check_fitted()
        _, _, _, evaluation_dataset = self._get_evaluation_dataset(dataset)

        if evaluation_dataset == "holdout":
            X_evaluation_raw = self.X_holdout_raw_
            y_evaluation_raw = pd.Series(self.y_holdout, index=X_evaluation_raw.index)
        else:
            X_evaluation_raw = self.X_validation_raw_
            y_evaluation_raw = self.y_validation_raw_
        if evaluation_dataset == "holdout":
            X_evidence_train = self.X_modeling_raw_
            y_evidence_train = self.y_modeling_raw_
            groups_evidence_train = self.groups_modeling_
            groups_evaluation = self.groups_holdout_
        else:
            X_evidence_train = self.X_train_raw_
            y_evidence_train = self.y_train_raw_
            groups_evidence_train = self.groups_train_
            groups_evaluation = self.groups_validation_

        self.evidence_report_ = build_evidence_report(
            X=self.X_modeling_raw_,
            y=self.y_modeling_raw_,
            y_leakage=self.y_modeling_original_,
            X_train=X_evidence_train,
            y_train=y_evidence_train,
            X_evaluation=X_evaluation_raw,
            y_evaluation=y_evaluation_raw,
            selected_estimator=self.selected_estimator_,
            candidate_estimators=(
                self.raw_fitted_models_ if include_candidate_comparison else None
            ),
            metric_name=self.score_metric_name,
            binary=self.binary,
            preprocessor_factory=self._make_model_preprocessor,
            evaluation_dataset=evaluation_dataset,
            holdout_available=self.X_holdout is not None,
            groups=self.groups_modeling_,
            groups_train=groups_evidence_train,
            groups_evaluation=groups_evaluation,
            cv_splits=self.evidence_cv_splits,
            cv_repeats=self.evidence_cv_repeats,
            confidence_level=self.evidence_confidence_level,
            random_state=self.random_state,
            practical_margin=self.evidence_practical_margin,
        )
        self.validation_integrity_ = self.evidence_report_["validation_integrity"]
        self.leakage_checks_ = self.evidence_report_["leakage_checks"]
        self.baseline_comparison_ = self.evidence_report_["baseline_comparison"]
        self.score_stability_ = self.evidence_report_["score_stability"]
        self.selection_guidance_ = self.evidence_report_["selection_guidance"]
        return self.evidence_report_

    def _make_model_preprocessor(self, model_name: Optional[str] = None):
        if not self.preprocess:
            return None
        profile = self._preprocessing_profile_for_label(model_name)
        kwargs = self.preprocessor_kwargs.copy()
        if profile == "native_categorical" and (
            kwargs.get("pca") or kwargs.get("feature_selection")
        ):
            profile = "tree_ohe"
        if all(
            [
                profile == "native_categorical",
                self.imbalanced_,
                kwargs.get("imbalanced_resampling", True),
            ]
        ):
            profile = "tree_ohe"
        kwargs["profile"] = profile
        return Preprocessor(**kwargs)

    def _preprocessing_profile_for_label(self, model_name: Optional[str]) -> str:
        if self.preprocessing_profile == "generic_ohe" or model_name is None:
            return "generic_ohe"

        normalized_name = self._normalize_model_label(model_name)
        if normalized_name in set(available_model_names()):
            return preprocessing_profile_for_model(normalized_name)
        return "generic_ohe"

    @staticmethod
    def _normalize_model_label(model_name: str) -> str:
        for prefix in ("MAMUT Candidate (", "MAMUT Selected ("):
            if model_name.startswith(prefix) and model_name.endswith(")"):
                return model_name[len(prefix) : -1]
        baseline_aliases = {
            "Logistic Regression": "LogisticRegression",
            "Random Forest": "RandomForestClassifier",
            "Dummy Most Frequent": "DummyClassifier",
        }
        return baseline_aliases.get(model_name, model_name)

    def _make_evidence_preprocessor(self, model_name: Optional[str] = None):
        return self._make_model_preprocessor(model_name)

    def _get_evaluation_dataset(self, dataset: str):
        if dataset not in {"auto", "validation", "holdout"}:
            raise ValueError("dataset must be one of: 'auto', 'validation', 'holdout'.")

        if dataset == "auto":
            dataset = "holdout" if self.X_holdout is not None else "validation"

        if dataset == "holdout":
            if self.X_holdout is None or self.y_holdout is None:
                raise ValueError(
                    "No holdout data is available. Provide holdout_size, or pass "
                    "X_holdout and y_holdout to fit()."
                )
            if self.holdout_summary_ is None:
                self.holdout_summary_ = self._score_models_on_dataset(
                    self.raw_fitted_models_,
                    self.X_holdout_raw_,
                    self.y_holdout,
                )
            return (
                self.X_holdout,
                self.y_holdout,
                self.holdout_summary_,
                "holdout",
            )

        return (
            self.X_validation,
            self.y_validation,
            self.validation_summary_,
            "validation",
        )

    def save_best_model(self, path: str) -> None:
        """
        Saves the best model to the specified path.

        Parameters
        ----------
        path : str
            The path to save the best model.
        """
        self._check_fitted()
        save_path = os.path.join(
            path, f"{self._pipeline_model_name(self.best_model_)}.joblib"
        )
        joblib.dump(self.best_model_, save_path)
        log.info(f"Saved best model to {save_path}")

    def create_ensemble(self, voting: Literal["soft", "hard"] = "soft") -> Pipeline:
        """
        Creates an ensemble of the fitted models.

        Parameters
        ----------
        voting : Literal["soft", "hard"]
            Voting strategy for the ensemble.

        Returns
        -------
        Pipeline
            The ensemble model pipeline.
        """
        self._check_fitted()

        ensemble = VotingClassifier(
            estimators=self._ensemble_estimators(),
            voting=voting,
        )
        ensemble.fit(self.X_train_raw_, self.y_train_raw_)
        y_pred = ensemble.predict(self.X_validation_raw_)
        score = self.score_metric(self.y_validation_raw_, y_pred)

        self.ensemble_ = self._make_public_pipeline(ensemble, None)
        log.info(
            f"Created ensemble with all models and voting='{voting}'. "
            f"Ensemble score on validation set: {score:.4f} {self.score_metric.__name__}"
        )

        return self.ensemble_

    def _create_greedy_ensemble_voting(
        self, n_models: int = 6, voting: Literal["soft", "hard"] = "soft"
    ) -> Pipeline:
        """
        Creates a greedy ensemble of the fitted models.

        Parameters
        ----------
        n_models : int
            Number of models to include in the ensemble.
        voting : Literal["soft", "hard"]
            Voting strategy for the ensemble.

        Returns
        -------
        Pipeline
            The greedy ensemble model pipeline.
        """
        self._check_fitted()
        n_models = min(n_models, len(self.raw_fitted_models_))
        if n_models < 2:
            raise ValueError(
                "At least two fitted models are required to build an ensemble."
            )
        ranked_names = (
            self.validation_summary_.sort_values(
                self.score_metric_name, ascending=False
            )["model"]
            .head(n_models)
            .tolist()
        )
        selected_names = [ranked_names[0]]
        best_ensemble = None
        best_score = -np.inf

        while len(selected_names) < n_models:
            round_best_name = None
            round_best_ensemble = None
            round_best_score = -np.inf
            for candidate_name in sorted(set(ranked_names).difference(selected_names)):
                candidate_names = [*selected_names, candidate_name]
                candidate_ensemble = VotingClassifier(
                    estimators=self._ensemble_estimators(candidate_names),
                    voting=voting,
                )
                candidate_ensemble.fit(self.X_train_raw_, self.y_train_raw_)
                score = self.score_metric(
                    self.y_validation_raw_,
                    candidate_ensemble.predict(self.X_validation_raw_),
                )
                if score > round_best_score:
                    round_best_name = candidate_name
                    round_best_ensemble = candidate_ensemble
                    round_best_score = score
            selected_names.append(round_best_name)
            best_ensemble = round_best_ensemble
            best_score = round_best_score

        self.ensemble_models_ = selected_names
        self.greedy_ensemble_ = self._make_public_pipeline(best_ensemble, None)

        log.info(
            f"Created greedy ensemble with voting='{voting}' \n"
            f"and {n_models} models: {selected_names} \n"
            f"Ensemble score on validation set: {best_score:.4f} {self.score_metric.__name__}"
        )

        return self.greedy_ensemble_

    def create_greedy_ensemble(self, max_models=6):
        return self._create_greedy_ensemble_voting(
            n_models=max_models,
            voting="soft",
        )

    def _predict(self, X: pd.DataFrame, proba: bool = False):
        """
        Predicts the target variable or probabilities for the given data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        proba : bool
            Whether to predict probabilities instead of the target variable.

        Returns
        -------
        np.ndarray
            Predicted target variable or probabilities.
        """
        self._check_fitted()
        if proba:
            return self.best_model_.predict_proba(X)
        return self.best_model_.predict(X)

    def _check_fitted(self):
        """
        Checks if the model has been fitted.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.best_model_:
            raise RuntimeError(
                "Can't predict because no model has been fitted. "
                "Please call fit() method first."
            )

    @staticmethod
    def _check_categorical(y):
        """
        Checks if the target variable is categorical.

        Parameters
        ----------
        y : pd.Series
            The target variable.

        Raises
        ------
        ValueError
            If the target variable is not categorical.
        """
        if pd.api.types.is_float_dtype(y):
            raise ValueError("Target variable must be categorical.")

    @staticmethod
    def _validate_split_size(value: float, name: str) -> None:
        if not 0 < value < 1:
            raise ValueError(f"{name} must be greater than 0 and less than 1.")
