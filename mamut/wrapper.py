import logging
import os
import time
from copy import copy
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from mamut.preprocessing.preprocessing import Preprocessor
from mamut.utils.utils import metric_dict

from .evaluation import ModelEvaluator
from .model_selection import ModelSelector

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target variable.
    y_test : pd.Series
        Test target variable.
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
            "roc_auc",
        ] = "f1",
        optimization_method: Literal["random_search", "bayes"] = "bayes",
        n_iterations: Optional[int] = 30,
        random_state: Optional[int] = 42,
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
        score_metric : Literal["accuracy", "precision", "recall", "f1", "balanced_accuracy", "jaccard", "roc_auc"]
            Metric used to evaluate model performance.
        optimization_method : Literal["random_search", "bayes"]
            Method for hyperparameter optimization.
        n_iterations : Optional[int]
            Number of iterations for optimization.
        random_state : Optional[int]
            Random state for reproducibility.
        **preprocessor_kwargs
            Additional keyword arguments for the Preprocessor.
        """
        self.preprocess = preprocess
        self.imb_threshold = imb_threshold
        self.exclude_models = exclude_models
        self.score_metric = metric_dict[score_metric]
        self.optimization_method = optimization_method
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.preprocessor = Preprocessor(**preprocessor_kwargs) if preprocess else None
        self.le = LabelEncoder()
        self.model_selector = None

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.binary = None
        self.roc = None

        self.raw_fitted_models_ = None
        self.fitted_models_ = None
        self.best_model_ = None

        self.best_score_ = None
        self.training_summary_ = None
        self.optuna_studies_ = None

        self.ensemble_ = None
        self.greedy_ensemble_ = None
        self.ensemble_models_ = None
        self.imbalanced_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the model to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series
            The target variable.

        Returns
        -------
        Pipeline
            The best model pipeline.
        """
        Mamut._check_categorical(y)
        if y.value_counts(normalize=True).min() < self.imb_threshold:
            self.imbalanced_ = True

        y = self.le.fit_transform(y)
        y = pd.Series(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        if self.preprocess:
            X_train, y_train = self.preprocessor.fit_transform(X_train, y_train)
            X_test = self.preprocessor.transform(X_test)

        self.X_train = X_train  # np.ndarray
        self.X_test = X_test  # np.ndarray
        self.y_train = y_train  # np.ndarray
        self.y_test = y_test  # pd.Series
        self.X = X
        self.y = y

        self.model_selector = ModelSelector(
            X_train,
            y_train,
            X_test,
            y_test,
            exclude_models=self.exclude_models,
            score_metric=self.score_metric,
            optimization_method=self.optimization_method,
            n_iterations=self.n_iterations,
            random_state=self.random_state,
        )

        (
            best_model,
            params_for_best_model,
            score_for_best_model,
            fitted_models,
            training_summary,
            studies,
        ) = self.model_selector.compare_models()

        self.raw_fitted_models_ = fitted_models
        self.optuna_studies_ = studies
        self.fitted_models_ = [
            Pipeline([("preprocessor", self.preprocessor), ("model", model)])
            for model in fitted_models.values()
        ]

        # Update the score metric based on binary/multiclass problem (for ensembles)
        self.score_metric = self.model_selector.score_metric
        self.score_metric_name = self.model_selector.score_metric_name
        self.binary = self.model_selector.binary
        self.roc = self.model_selector.roc
        self.best_score_ = score_for_best_model
        self.best_model_ = Pipeline(
            [("preprocessor", self.preprocessor), ("model", best_model)]
        )
        self.training_summary_ = training_summary

        log.info(f"Best model: {best_model.__class__.__name__}")

        # Models_dir with time signature
        cwd = os.getcwd()
        models_dir = os.path.join(
            cwd,
            "fitted_models",
            str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())),
        )
        os.makedirs(models_dir, exist_ok=True)
        for model in self.fitted_models_:
            model_name = model.named_steps["model"].__class__.__name__
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            log.info(f"Saved model {model_name} to {model_path}")

        return self.best_model_

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

    def evaluate(self, n_top_models: int = 3) -> None:
        """
        Evaluates the fitted models.
        """
        self._check_fitted()

        evaluator = ModelEvaluator(
            self.raw_fitted_models_,
            X_test=self.X_test,
            y_test=self.y_test,
            X_train=self.X_train,
            y_train=self.y_train,
            X=self.X,
            y=self.y,
            optimizer=self.optimization_method,
            metric=self.score_metric_name,
            n_trials=self.n_iterations,
            excluded_models=self.exclude_models,
            studies=self.optuna_studies_,
            training_summary=self.training_summary_,
            pca_loadings=self.preprocessor.pca_loadings_,
            binary=self.model_selector.binary,
            preprocessing_steps=self.preprocessor.report(),
            n_top_models=n_top_models,
            is_ensemble=self.greedy_ensemble_ is not None,
            greedy_ensemble=self.greedy_ensemble_,
        )

        evaluator.evaluate_to_html(self.training_summary_)
        evaluator.plot_results_in_notebook()

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
            path, f"{self.best_model_.named_steps['model'].__class__.__name__}.joblib"
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
                    model.named_steps["model"].__class__.__name__,
                    clone(model.named_steps["model"]),
                )
                for model in self.fitted_models_
            ],
            voting=voting,
        )
        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_test)
        score = self.score_metric(self.y_test.values, y_pred)

        self.ensemble_ = Pipeline(
            [("preprocessor", self.preprocessor), ("model", ensemble)]
        )
        log.info(
            f"Created ensemble with all models and voting='{voting}'. "
            f"Ensemble score on test set: {score:.4f} {self.score_metric.__name__}"
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
        ensemble_models = [self.best_model_.named_steps["model"]]
        ensemble_scores = [self.best_score_]

        for _ in range(n_models - 1):
            best_score = 0
            best_model = None

            for model in self.fitted_models_:
                candidate_ensemble = ensemble_models + [model.named_steps["model"]]
                candidate_voting_clf = VotingClassifier(
                    estimators=[
                        (f"model_{i}", clone(m))
                        for i, m in enumerate(candidate_ensemble)
                    ],
                    voting=voting,
                )
                candidate_voting_clf.fit(self.X_train, self.y_train)
                score = self.score_metric(
                    self.y_test, candidate_voting_clf.predict(self.X_test)
                )

                if score > best_score:
                    best_score = score
                    best_model = model.named_steps["model"]

            ensemble_models.append(best_model)
            ensemble_scores.append(best_score)

        ensemble = VotingClassifier(
            estimators=[
                (f"model_{i}", clone(m)) for i, m in enumerate(ensemble_models)
            ],
            voting=voting,
        )
        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_test)
        score = self.score_metric(self.y_test, y_pred)

        self.ensemble_models_ = ensemble_models
        self.greedy_ensemble_ = Pipeline(
            [("preprocessor", self.preprocessor), ("model", ensemble)]
        )

        log.info(
            f"Created greedy ensemble with voting='{voting}' \n"
            f"and {n_models} models: {[m.__class__.__name__ for m in ensemble_models]} \n"
            f"Ensemble score on test set: {score:.4f} {self.score_metric.__name__}"
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

        # Sort models list by their performance on X_test
        sorted_models = sorted(
            models, key=lambda model: self._score_model_on_test(model), reverse=True
        )

        # Start with the best and second best model
        ensemble_models = [sorted_models[0], sorted_models[1]]
        remaining_models = {
            model.__class__.__name__: model for model in copy(sorted_models[2:])
        }
        best_score = self._score_model_on_test(
            self._create_stacking_classifier(ensemble_models).fit(
                self.X_train, self.y_train
            )
        )
        ensemble_scores = [best_score]

        # Greedily add models to the ensemble
        for _ in range(max_models - 2):
            best_score = 0
            best_model = None

            for model in remaining_models.values():
                candidate_ensemble = ensemble_models + [model]
                candidate_stacking_clf = self._create_stacking_classifier(
                    candidate_ensemble
                )
                candidate_stacking_clf.fit(self.X_train, self.y_train)
                score = self._score_model_on_test(candidate_stacking_clf)

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
        self.greedy_ensemble_ = Pipeline(
            [("preprocessor", self.preprocessor), ("model", final_stacking_clf)]
        )

        return self.greedy_ensemble_

    def _score_model_on_test(self, model):
        if self.roc:
            if self.binary:
                score_on_test = self.score_metric(
                    self.y_test.values, model.predict_proba(self.X_test)[:, 1]
                )
            else:
                score_on_test = self.score_metric(
                    self.y_test.values, model.predict_proba(self.X_test)
                )
        else:
            score_on_test = self.score_metric(
                self.y_test.values, model.predict(self.X_test)
            )
        return score_on_test

    def _create_stacking_classifier(self, models):
        estimators = [(model.__class__.__name__, clone(model)) for model in models]
        return StackingClassifier(
            estimators=estimators, final_estimator=RandomForestClassifier()
        )

    def _calculate_disagreement(self, model1, model2, X_test):
        #  TODO: THIS IS WORK IN PROGRESS... DO NOT USE
        """Calculate disagreement between two models' predictions."""
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        return np.mean(pred1 != pred2)

    def _ensemble_selection(
        self, max_ensemble_size=5, voting: Literal["soft", "hard"] = "soft"
    ):
        """Greedy algorithm to select the best subset of models for ensemble."""
        #  TODO: THIS IS WORK IN PROGRESS... DO NOT USE
        models = [
            (model.__class__.__name__, model)
            for model in self.raw_fitted_models_.values()
        ]
        print(models)
        selected_models = []
        remaining_models = models.copy()
        best_score = 0
        ensemble_performance = []

        # Initialize with the best performing model on test set
        scores = {
            name: self.score_metric(self.y_test, model.predict(self.X_test))
            for name, model in models
        }
        print("Scores:", scores)
        best_model_name, best_model = max(scores.items(), key=lambda item: item[1])
        print("Best model:", best_model_name, "with score:", best_model)
        selected_models.append((best_model_name, best_model))
        remaining_models.remove((best_model_name, best_model))
        best_score = scores[best_model_name]
        ensemble_performance.append(best_score)

        print(f"Starting with best model: {best_model_name} with score: {best_score}")

        # Greedily add models based on performance and diversity
        while len(selected_models) < max_ensemble_size and remaining_models:
            best_model_to_add = None
            best_new_score = best_score
            for name, model in remaining_models:
                # Test current ensemble with this model added
                current_ensemble = VotingClassifier(
                    estimators=selected_models + [(name, model)], voting=voting
                )
                current_ensemble.fit(self.X_train, self.y_train)
                ensemble_score = self.score_metric(
                    self.y_test, current_ensemble.predict(self.X_test)
                )

                # Compute diversity with selected models
                diversity = np.mean(
                    [
                        self._calculate_disagreement(
                            model, selected_model[1], self.X_test
                        )
                        for selected_model in selected_models
                    ]
                )

                # Score considering both accuracy improvement and diversity
                weighted_score = (
                    ensemble_score + 0.1 * diversity
                )  # 0.1 is a diversity weight factor
                if weighted_score > best_new_score:
                    best_model_to_add = (name, model)
                    best_new_score = weighted_score

            if best_model_to_add:
                selected_models.append(best_model_to_add)
                remaining_models.remove(best_model_to_add)
                best_score = best_new_score
                ensemble_performance.append(best_new_score)
                print(
                    f"Added {best_model_to_add[0]} to ensemble, new weighted score: {best_new_score}"
                )
            else:
                break  # No improvement

        # Final ensemble
        final_ensemble = VotingClassifier(estimators=selected_models, voting=voting)
        final_ensemble.fit(self.X_train, self.y_train)
        return final_ensemble, ensemble_performance

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
