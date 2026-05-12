import logging
import os
import time
from copy import copy
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from mamut.evidence import build_evidence_report
from mamut.preprocessing.preprocessing import Preprocessor
from mamut.utils.utils import metric_dict, model_param_dict

from .evaluation import ModelEvaluator
from .model_selection import ModelSelector

log = logging.getLogger(__name__)


class LabelDecodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap an encoded-label classifier so public predictions use original labels."""

    def __init__(self, estimator, label_encoder):
        self.estimator = estimator
        self.label_encoder = label_encoder

    def fit(self, X, y):
        y_encoded = self.label_encoder.transform(y)
        self.estimator.fit(X, y_encoded)
        return self

    def predict(self, X):
        encoded_predictions = self.estimator.predict(X)
        return self.label_encoder.inverse_transform(
            np.asarray(encoded_predictions).astype(int)
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
        score_metric: Literal[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "jaccard",
            "roc_auc_score",
        ] = "f1",
        optimization_method: Literal["random_search", "bayes"] = "bayes",
        n_iterations: int = 30,
        random_state: Optional[int] = 42,
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
        score_metric : Literal["accuracy", "precision", "recall", "f1", "balanced_accuracy", "jaccard", "roc_auc_score"]
            Metric used to evaluate model performance.
        optimization_method : Literal["random_search", "bayes"]
            Method for hyperparameter optimization.
        n_iterations : Optional[int]
            Number of iterations for optimization.
        random_state : Optional[int]
            Random state for reproducibility.
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
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError(
                "n_iterations must be an integer greater than or equal to 1."
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
        exclude_models = exclude_models or []
        unknown_models = sorted(set(exclude_models) - set(model_param_dict))
        if unknown_models:
            valid_models = ", ".join(model_param_dict)
            raise ValueError(
                f"exclude_models contains unsupported model names: {unknown_models}. "
                f"Valid model names are: {valid_models}."
            )
        if len(exclude_models) >= len(model_param_dict):
            raise ValueError("exclude_models cannot remove every supported model.")

        self.preprocess = preprocess
        self.imb_threshold = imb_threshold
        self.exclude_models = exclude_models
        self.score_metric = metric_dict[score_metric]
        self.score_metric_name = self.score_metric.__name__
        self.optimization_method = optimization_method
        self.n_iterations = n_iterations
        self.random_state = random_state
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

        self.preprocessor = Preprocessor(**preprocessor_kwargs) if preprocess else None
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
        self.X_test = None
        self.y_test = None
        self.binary = None
        self.roc = None

        self.raw_fitted_models_ = None
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
        self.optuna_studies_ = None
        self.models_output_path_ = None
        self.evidence_report_ = None
        self.validation_integrity_ = None
        self.leakage_checks_ = None
        self.baseline_comparison_ = None
        self.score_stability_ = None
        self.selection_guidance_ = None
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

        X_modeling = X.copy()
        y_modeling = y_encoded.copy()
        y_modeling_original = y_original.copy()

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
        elif self.holdout_size is not None:
            (
                X_modeling,
                self.X_holdout_raw_,
                y_modeling,
                y_holdout_encoded,
                y_modeling_original,
                self.y_holdout_original_,
            ) = train_test_split(
                X_modeling,
                y_modeling,
                y_modeling_original,
                test_size=self.holdout_size,
                stratify=y_modeling,
                random_state=self.random_state,
            )
        else:
            y_holdout_encoded = None

        self.X_modeling_raw_ = X_modeling.copy()
        self.y_modeling_raw_ = y_modeling.copy()
        self.y_modeling_original_ = y_modeling_original.copy()

        (
            X_train_raw,
            X_validation_raw,
            y_train,
            y_validation,
        ) = train_test_split(
            X_modeling,
            y_modeling,
            test_size=self.validation_size,
            stratify=y_modeling,
            random_state=self.random_state,
        )

        X_train = X_train_raw
        X_validation = X_validation_raw
        self.y_train_raw_ = y_train.copy()
        self.y_validation_raw_ = y_validation.copy()
        if self.preprocess:
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
            exclude_models=self.exclude_models,
            score_metric=self.score_metric,
            optimization_method=self.optimization_method,
            n_iterations=self.n_iterations,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        (
            best_model,
            _,
            score_for_best_model,
            fitted_models,
            validation_summary,
            studies,
        ) = self.model_selector.compare_models()

        self.raw_fitted_models_ = fitted_models
        self.optuna_studies_ = studies
        self.fitted_models_ = [
            self._make_public_pipeline(model, self.preprocessor)
            for model in fitted_models.values()
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
        self.best_score_ = score_for_best_model
        self.validation_selected_estimator_ = best_model
        self.selected_estimator_ = best_model
        public_preprocessor = self.preprocessor
        if self.refit_final_model:
            self.final_preprocessor_, self.final_estimator_ = (
                self._refit_selected_model(best_model)
            )
            self.selected_estimator_ = self.final_estimator_
            public_preprocessor = self.final_preprocessor_

        self.best_model_ = self._make_public_pipeline(
            self.selected_estimator_, public_preprocessor
        )
        self.validation_summary_ = validation_summary
        self.training_summary_ = validation_summary
        self.holdout_summary_ = (
            self._score_models_on_dataset(
                self.raw_fitted_models_, self.X_holdout, self.y_holdout
            )
            if self.X_holdout is not None
            else None
        )
        self.holdout_score_ = (
            self._score_selected_model_on_holdout()
            if self.X_holdout is not None
            else None
        )

        log.info(f"Best model: {best_model.__class__.__name__}")

        if self.save_models:
            self._save_fitted_models()

        return self.best_model_

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
        preprocessor = (
            Preprocessor(**self.preprocessor_kwargs) if self.preprocess else None
        )
        if preprocessor is not None:
            X_final, y_final = preprocessor.fit_transform(
                self.X_modeling_raw_.copy(),
                self.y_modeling_raw_.copy(),
            )
        else:
            X_final = self._as_model_input(self.X_modeling_raw_)
            y_final = np.asarray(self.y_modeling_raw_)

        final_estimator = clone(selected_estimator)
        final_estimator.fit(X_final, y_final)
        return preprocessor, final_estimator

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
            self.selected_estimator_, self.X_holdout, self.y_holdout
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
            rows.append(
                {
                    "model": model_name,
                    **self._score_model_with_metrics(model, X, y),
                    "duration": duration_by_model.get(model_name, np.nan),
                }
            )

        return pd.DataFrame(rows)

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
        X_evaluation, y_evaluation, evaluation_summary, evaluation_dataset = (
            self._get_evaluation_dataset(dataset)
        )
        evidence_report = (
            self.generate_evidence(dataset=evaluation_dataset)
            if include_evidence
            else None
        )

        evaluator = ModelEvaluator(
            self.raw_fitted_models_,
            X_evaluation=X_evaluation,
            y_evaluation=y_evaluation,
            X_train=self.X_train,
            y_train=self.y_train,
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
    ) -> dict:
        self._check_fitted()
        _, _, _, evaluation_dataset = self._get_evaluation_dataset(dataset)

        if evaluation_dataset == "holdout":
            X_evaluation_raw = self.X_holdout_raw_
            y_evaluation_raw = pd.Series(self.y_holdout, index=X_evaluation_raw.index)
        else:
            X_evaluation_raw = self.X_validation_raw_
            y_evaluation_raw = self.y_validation_raw_

        self.evidence_report_ = build_evidence_report(
            X=self.X_modeling_raw_,
            y=self.y_modeling_raw_,
            y_leakage=self.y_modeling_original_,
            X_train=self.X_train_raw_,
            y_train=self.y_train_raw_,
            X_evaluation=X_evaluation_raw,
            y_evaluation=y_evaluation_raw,
            selected_estimator=self.selected_estimator_,
            metric_name=self.score_metric_name,
            binary=self.binary,
            preprocessor_factory=self._make_evidence_preprocessor,
            evaluation_dataset=evaluation_dataset,
            holdout_available=self.X_holdout is not None,
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

    def _make_evidence_preprocessor(self):
        if not self.preprocess:
            return None
        return Preprocessor(**self.preprocessor_kwargs)

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
                    self.X_holdout,
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
            estimators=[
                (
                    model_name,
                    clone(model),
                )
                for model_name, model in self.raw_fitted_models_.items()
            ],
            voting=voting,
        )
        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_validation)
        score = self.score_metric(self.y_validation, y_pred)

        self.ensemble_ = self._make_public_pipeline(ensemble, self.preprocessor)
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

        # Initialize the ensemble with the best model
        ensemble_models = [self.validation_selected_estimator_]
        ensemble_scores = [self.best_score_]

        for _ in range(n_models - 1):
            best_score = -np.inf
            best_model = None

            for model in self.raw_fitted_models_.values():
                candidate_ensemble = ensemble_models + [model]
                candidate_voting_clf = VotingClassifier(
                    estimators=[
                        (f"model_{i}", clone(m))
                        for i, m in enumerate(candidate_ensemble)
                    ],
                    voting=voting,
                )
                candidate_voting_clf.fit(self.X_train, self.y_train)
                score = self.score_metric(
                    self.y_validation,
                    candidate_voting_clf.predict(self.X_validation),
                )

                if score > best_score:
                    best_score = score
                    best_model = model

            ensemble_models.append(best_model)
            ensemble_scores.append(best_score)

        ensemble = VotingClassifier(
            estimators=[
                (f"model_{i}", clone(m)) for i, m in enumerate(ensemble_models)
            ],
            voting=voting,
        )
        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_validation)
        score = self.score_metric(self.y_validation, y_pred)

        self.ensemble_models_ = ensemble_models
        self.greedy_ensemble_ = self._make_public_pipeline(ensemble, self.preprocessor)

        log.info(
            f"Created greedy ensemble with voting='{voting}' \n"
            f"and {n_models} models: {[m.__class__.__name__ for m in ensemble_models]} \n"
            f"Ensemble score on validation set: {score:.4f} {self.score_metric.__name__}"
        )

        return self.greedy_ensemble_

    def create_greedy_ensemble(self, max_models=6):
        self._check_fitted()
        models = [model for name, model in self.raw_fitted_models_.items()]

        if max_models > len(models):
            max_models = len(models)
            log.info(
                f"Max models set to {max_models} as there are only {len(models)} models available"
                f"in the bag-of-models used in this experiment."
            )

        if len(models) < 2:
            raise ValueError(
                "At least two fitted models are required to build an ensemble."
            )

        # Sort models list by their performance on the validation set.
        sorted_models = sorted(
            models,
            key=lambda model: self._score_model_on_validation(model),
            reverse=True,
        )

        # Start with the best and second best model
        ensemble_models = [sorted_models[0], sorted_models[1]]
        remaining_models = {
            model.__class__.__name__: model for model in copy(sorted_models[2:])
        }
        best_score = self._score_model_on_validation(
            self._create_stacking_classifier(ensemble_models).fit(
                self.X_train, self.y_train
            )
        )
        ensemble_scores = [best_score]

        # Greedily add models to the ensemble
        for _ in range(max_models - 2):
            best_score = -np.inf
            best_model = None

            for model in remaining_models.values():
                candidate_ensemble = ensemble_models + [model]
                candidate_stacking_clf = self._create_stacking_classifier(
                    candidate_ensemble
                )
                candidate_stacking_clf.fit(self.X_train, self.y_train)
                score = self._score_model_on_validation(candidate_stacking_clf)

                if score > best_score:
                    best_score = score
                    best_model = model

            ensemble_models.append(best_model)
            ensemble_scores.append(best_score)
            # Remove this best model from the remaining models dict
            del remaining_models[best_model.__class__.__name__]

        # From nested family of ensembles pick the best one based on ensemble_scores
        best_score = max(ensemble_scores)
        # Check from the end of the list to find the best ensemble
        for i in range(len(ensemble_scores) - 1, 0, -1):
            if ensemble_scores[i] == best_score:
                ensemble_models = ensemble_models[: i + 2]
                break

        # Create the final stacking classifier
        final_stacking_clf = self._create_stacking_classifier(ensemble_models)
        final_stacking_clf.fit(self.X_train, self.y_train)
        self.greedy_ensemble_ = final_stacking_clf

        log.info(
            f"Created greedy ensemble with {len(ensemble_models)} models. Best score: {best_score:.4f}."
            f"For details on the ensemble please run evaluate() method and see the report."
        )

        # Create a pipeline with the best ensemble
        self.greedy_ensemble_ = self._make_public_pipeline(
            final_stacking_clf, self.preprocessor
        )

        return self.greedy_ensemble_

    def _score_model_on_validation(self, model):
        if self.roc:
            if self.binary:
                score_on_validation = self.score_metric(
                    self.y_validation,
                    model.predict_proba(self.X_validation)[:, 1],
                )
            else:
                score_on_validation = self.score_metric(
                    self.y_validation,
                    model.predict_proba(self.X_validation),
                )
        else:
            score_on_validation = self.score_metric(
                self.y_validation, model.predict(self.X_validation)
            )
        return score_on_validation

    def _create_stacking_classifier(self, models):
        estimators = [(model.__class__.__name__, clone(model)) for model in models]
        return StackingClassifier(
            estimators=estimators, final_estimator=RandomForestClassifier()
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
