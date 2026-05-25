import ast
import inspect
import logging
import time
import warnings
from copy import copy
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier  # noqa
from lightgbm import LGBMClassifier  # noqa
from optuna.samplers import RandomSampler, TPESampler
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (  # noqa
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB  # noqa
from sklearn.neighbors import KNeighborsClassifier  # noqa
from sklearn.neural_network import MLPClassifier  # noqa
from sklearn.svm import SVC  # noqa
from xgboost import XGBClassifier  # noqa

from mamut.preprocessing.preprocessing import Preprocessor
from mamut.utils.utils import (
    SEARCH_PROFILES,
    adjust_search_spaces,
    model_names_for_profile,
    model_param_dict,
    sample_parameter,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateModel:
    name: str
    estimator_factory: Callable[[Optional[int], Optional[int]], object]
    runtime_cost: Literal["low", "medium", "high"] = "medium"
    supports_predict_proba: bool = True
    preprocessing_profile: Literal["generic_ohe", "tree_ohe", "native_categorical"] = (
        "generic_ohe"
    )

    def create(self, *, random_state: Optional[int], n_jobs: Optional[int]):
        return self.estimator_factory(random_state, n_jobs)


def _sklearn_factory(model_class, **default_params):
    def factory(random_state: Optional[int], n_jobs: Optional[int]):
        params = default_params.copy()
        valid_params = model_class().get_params()
        if random_state is not None and "random_state" in valid_params:
            params.setdefault("random_state", random_state)
        if n_jobs is not None and "n_jobs" in valid_params:
            params.setdefault("n_jobs", n_jobs)
        return model_class(**params)

    return factory


def _xgboost_factory(random_state: Optional[int], n_jobs: Optional[int]):
    params = {
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    if random_state is not None:
        params["random_state"] = random_state
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    return XGBClassifier(**params)


def _lightgbm_factory(random_state: Optional[int], n_jobs: Optional[int]):
    params = {
        "verbosity": -1,
    }
    if random_state is not None:
        params["random_state"] = random_state
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    return LGBMClassifier(**params)


def _catboost_factory(random_state: Optional[int], n_jobs: Optional[int]):
    params = {
        "allow_writing_files": False,
        "verbose": False,
    }
    if random_state is not None:
        params["random_seed"] = random_state
    if n_jobs is not None:
        params["thread_count"] = n_jobs
    return CatBoostClassifier(**params)


MODEL_REGISTRY = {
    "LogisticRegression": CandidateModel(
        "LogisticRegression", _sklearn_factory(LogisticRegression), runtime_cost="low"
    ),
    "RandomForestClassifier": CandidateModel(
        "RandomForestClassifier",
        _sklearn_factory(RandomForestClassifier),
        runtime_cost="medium",
        preprocessing_profile="tree_ohe",
    ),
    "ExtraTreesClassifier": CandidateModel(
        "ExtraTreesClassifier",
        _sklearn_factory(ExtraTreesClassifier),
        runtime_cost="medium",
        preprocessing_profile="tree_ohe",
    ),
    "HistGradientBoostingClassifier": CandidateModel(
        "HistGradientBoostingClassifier",
        _sklearn_factory(HistGradientBoostingClassifier),
        runtime_cost="medium",
        preprocessing_profile="tree_ohe",
    ),
    "SVC": CandidateModel("SVC", _sklearn_factory(SVC), runtime_cost="high"),
    "XGBClassifier": CandidateModel(
        "XGBClassifier",
        _xgboost_factory,
        runtime_cost="high",
        preprocessing_profile="tree_ohe",
    ),
    "LGBMClassifier": CandidateModel(
        "LGBMClassifier",
        _lightgbm_factory,
        runtime_cost="medium",
        preprocessing_profile="native_categorical",
    ),
    "CatBoostClassifier": CandidateModel(
        "CatBoostClassifier",
        _catboost_factory,
        runtime_cost="high",
        preprocessing_profile="native_categorical",
    ),
    "MLPClassifier": CandidateModel(
        "MLPClassifier", _sklearn_factory(MLPClassifier), runtime_cost="high"
    ),
    "GaussianNB": CandidateModel(
        "GaussianNB", _sklearn_factory(GaussianNB), runtime_cost="low"
    ),
    "KNeighborsClassifier": CandidateModel(
        "KNeighborsClassifier",
        _sklearn_factory(KNeighborsClassifier),
        runtime_cost="medium",
    ),
}


def available_model_names() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def preprocessing_profile_for_model(
    model_name: str,
) -> Literal["generic_ohe", "tree_ohe", "native_categorical"]:
    return MODEL_REGISTRY[model_name].preprocessing_profile


def categorical_feature_names(X) -> list:
    if not isinstance(X, pd.DataFrame):
        return []
    return [
        column
        for column in X.columns
        if any(
            [
                isinstance(X[column].dtype, pd.CategoricalDtype),
                pd.api.types.is_object_dtype(X[column]),
                pd.api.types.is_string_dtype(X[column]),
                pd.api.types.is_bool_dtype(X[column]),
            ]
        )
    ]


def fit_estimator(estimator, X, y):
    y = np.asarray(y).ravel()
    if isinstance(estimator, CatBoostClassifier) and isinstance(X, pd.DataFrame):
        cat_features = categorical_feature_names(X)
        if cat_features:
            estimator.fit(X, y, cat_features=cat_features)
            return estimator
    estimator.fit(X, y)
    return estimator


def make_preprocessor(preprocessor_factory: Optional[Callable], model_name: str):
    if preprocessor_factory is None:
        return None
    signature = inspect.signature(preprocessor_factory)
    if not signature.parameters:
        return preprocessor_factory()
    return preprocessor_factory(model_name)


class CandidatePipelineClassifier(ClassifierMixin, BaseEstimator):
    """Raw-input classifier that owns model-specific preprocessing."""

    def __init__(
        self,
        estimator,
        preprocess: bool = True,
        profile: str = "generic_ohe",
        preprocessor_kwargs: Optional[dict] = None,
    ):
        self.estimator = estimator
        self.preprocess = preprocess
        self.profile = profile
        self.preprocessor_kwargs = preprocessor_kwargs

    def fit(self, X, y):
        kwargs = dict(self.preprocessor_kwargs or {})
        kwargs["profile"] = self.profile
        self.preprocessor_ = Preprocessor(**kwargs) if self.preprocess else None
        if self.preprocessor_ is not None:
            X_fit, y_fit = self.preprocessor_.fit_transform(
                pd.DataFrame(X).copy(), pd.Series(y).copy()
            )
        else:
            X_fit, y_fit = X, np.asarray(y)
        self.estimator_ = clone(self.estimator)
        fit_estimator(self.estimator_, X_fit, y_fit)
        self.classes_ = self.estimator_.classes_
        return self

    def _transform(self, X):
        if self.preprocessor_ is None:
            return X
        return self.preprocessor_.transform(pd.DataFrame(X).copy())

    def predict(self, X):
        return self.estimator_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._transform(X))


class ModelSelector:
    def __init__(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        score_metric: Callable,
        X_train_raw: Optional[pd.DataFrame] = None,
        y_train_raw: Optional[pd.Series] = None,
        X_validation_raw: Optional[pd.DataFrame] = None,
        y_validation_raw: Optional[pd.Series] = None,
        groups_train_raw: Optional[pd.Series] = None,
        preprocessor_factory: Optional[Callable] = None,
        exclude_models: Optional[List[str]] = None,
        include_models: Optional[List[str]] = None,
        search_profile: Literal["quick", "balanced", "thorough"] = "balanced",
        optimization_method: Literal["random_search", "bayes"] = "bayes",
        n_iterations: int = 50,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = 1,
        verbose: bool = False,
    ):

        self.X_train = X_train
        self.y_train = np.asarray(y_train)
        self.X_validation = X_validation
        self.y_validation = np.asarray(y_validation)
        self.X_train_raw = X_train_raw
        self.y_train_raw = pd.Series(y_train_raw) if y_train_raw is not None else None
        self.X_validation_raw = X_validation_raw
        self.y_validation_raw = (
            pd.Series(y_validation_raw) if y_validation_raw is not None else None
        )
        self.groups_train_raw = (
            pd.Series(groups_train_raw) if groups_train_raw is not None else None
        )
        self.preprocessor_factory = preprocessor_factory
        self.base_score_metric = score_metric
        self.search_profile = search_profile
        self.optimization_method = optimization_method
        self.n_jobs = n_jobs
        self.model_names = self._resolve_model_names(
            include_models=include_models,
            exclude_models=exclude_models,
            search_profile=search_profile,
        )
        self.models = [
            self._instantiate_model(
                model_name, random_state=random_state, n_jobs=n_jobs
            )
            for model_name in self.model_names
        ]
        self.n_classes_ = len(np.unique(y_train))
        self.binary = True if self.n_classes_ == 2 else False
        self.score_metric_name = copy(score_metric.__name__)
        self.roc = self.score_metric_name == "roc_auc_score"
        self.score_metric = lambda y_true, y_pred: self._compute_score(
            score_metric, y_true, y_pred
        )

        self.optuna_sampler = (
            TPESampler(seed=random_state)
            if optimization_method == "bayes"
            else RandomSampler(seed=random_state)
        )
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.random_state = random_state
        self.SKF_ = self._make_splitter(
            self._cv_y(), groups=self.groups_train_raw, requested_splits=5
        )

    @staticmethod
    def _resolve_model_names(
        *,
        include_models: Optional[List[str]],
        exclude_models: Optional[List[str]],
        search_profile: str,
    ) -> list[str]:
        if search_profile not in SEARCH_PROFILES:
            raise ValueError(
                "search_profile must be one of: 'quick', 'balanced', 'thorough'."
            )
        if include_models and exclude_models:
            raise ValueError("Use include_models or exclude_models, not both.")

        if include_models:
            candidate_names = list(include_models)
        elif exclude_models:
            candidate_names = list(available_model_names())
        else:
            candidate_names = list(model_names_for_profile(search_profile))
        unknown_models = sorted(set(candidate_names) - set(available_model_names()))
        if unknown_models:
            valid_models = ", ".join(available_model_names())
            raise ValueError(
                f"Selected model names are unsupported: {unknown_models}. "
                f"Valid model names are: {valid_models}."
            )
        if exclude_models:
            candidate_names = [
                model_name
                for model_name in candidate_names
                if model_name not in set(exclude_models)
            ]
        if not candidate_names:
            raise ValueError("At least one supported model must be selected.")
        return candidate_names

    @staticmethod
    def _instantiate_model(
        model_name: str, random_state: Optional[int], n_jobs: Optional[int]
    ):
        return MODEL_REGISTRY[model_name].create(
            random_state=random_state,
            n_jobs=n_jobs,
        )

    @staticmethod
    def _safe_index(data, idx):
        if hasattr(data, "iloc"):
            return data.iloc[idx]
        return data[idx]

    @staticmethod
    def _effective_cv_splits(y, requested_splits: int) -> int:
        class_counts = pd.Series(y).value_counts()
        if class_counts.empty:
            raise ValueError("Cannot build stratified folds without target values.")
        effective_splits = min(requested_splits, int(class_counts.min()))
        if effective_splits < 2:
            raise ValueError(
                "At least two samples are required in every class for model selection."
            )
        return effective_splits

    def _cv_y(self):
        return self.y_train_raw if self.y_train_raw is not None else self.y_train

    def _make_splitter(
        self, y, groups=None, requested_splits: int = 5, repeat: int = 0
    ):
        effective_splits = self._effective_cv_splits(y, requested_splits)
        seed = None if self.random_state is None else self.random_state + repeat
        if groups is not None:
            effective_splits = min(effective_splits, pd.Series(groups).nunique())
            if effective_splits < 2:
                raise ValueError("At least two groups are required for grouped CV.")
            return StratifiedGroupKFold(
                n_splits=effective_splits, shuffle=True, random_state=seed
            )
        return StratifiedKFold(
            n_splits=effective_splits, shuffle=True, random_state=seed
        )

    def _compute_score(self, score_metric, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if self.roc:
            if self.binary:
                return score_metric(y_true, y_pred)
            return score_metric(y_true, y_pred, multi_class="ovr", average="weighted")

        if self.score_metric_name in {
            "precision_score",
            "recall_score",
            "f1_score",
            "jaccard_score",
        }:
            return score_metric(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )

        return score_metric(y_true, y_pred)

    def objective(self, trial, model):
        model_name = model.__class__.__name__
        if model_name in model_param_dict:
            param_grid = model_param_dict[model_name]
        else:
            raise ValueError(f"Model {model_name} not supported")

        param = {
            param_name: sample_parameter(trial, param_name, value)
            for param_name, value in param_grid.items()
        }

        param = adjust_search_spaces(param, model)

        configured_model = clone(model).set_params(**param)

        cv_scores = []
        X_for_split = self.X_train_raw if self.X_train_raw is not None else self.X_train
        y_for_split = self._cv_y()
        for train_idx, val_idx in self.SKF_.split(
            X_for_split, y_for_split, self.groups_train_raw
        ):
            X_train_fold, y_train_fold, X_val_fold, y_val_fold = self._prepare_cv_fold(
                train_idx, val_idx, model_name
            )

            fold_model = clone(configured_model)
            fit_estimator(fold_model, X_train_fold, y_train_fold)
            val_pred = (
                fold_model.predict_proba(X_val_fold)
                if self.roc
                else fold_model.predict(X_val_fold)
            )
            if self.binary and self.roc:
                val_pred = val_pred[:, 1]

            cv_scores.append(self.score_metric(y_val_fold, val_pred))
        mean_cv_score = np.mean(cv_scores)

        return mean_cv_score

    def _prepare_cv_fold(self, train_idx, val_idx, model_name: str):
        if self.X_train_raw is None or self.preprocessor_factory is None:
            X_train_fold = self._safe_index(self.X_train, train_idx)
            X_val_fold = self._safe_index(self.X_train, val_idx)
            y_train_fold = self.y_train[train_idx]
            y_val_fold = self.y_train[val_idx]
            return X_train_fold, y_train_fold, X_val_fold, y_val_fold

        X_train_fold_raw = self.X_train_raw.iloc[train_idx]
        X_val_fold_raw = self.X_train_raw.iloc[val_idx]
        y_train_fold_raw = self.y_train_raw.iloc[train_idx]
        y_val_fold = self.y_train_raw.iloc[val_idx].to_numpy()

        preprocessor = make_preprocessor(self.preprocessor_factory, model_name)
        if preprocessor is None:
            return (
                self._as_model_input(X_train_fold_raw),
                y_train_fold_raw.to_numpy(),
                self._as_model_input(X_val_fold_raw),
                y_val_fold,
            )

        X_train_fold, y_train_fold = preprocessor.fit_transform(
            X_train_fold_raw.copy(), y_train_fold_raw.copy()
        )
        X_val_fold = preprocessor.transform(X_val_fold_raw.copy())
        return X_train_fold, np.asarray(y_train_fold), X_val_fold, y_val_fold

    def optimize_model(self, model):
        study = optuna.create_study(direction="maximize", sampler=self.optuna_sampler)
        start_time = time.time()
        study.optimize(
            lambda trial: self.objective(trial, model),
            n_trials=self.n_iterations,
            show_progress_bar=self.verbose,
        )
        end_time = time.time()
        duration = end_time - start_time
        best_params = study.best_params
        best_params = adjust_search_spaces(best_params, model)
        hidden_sizes = best_params.get("hidden_layer_sizes")
        if isinstance(hidden_sizes, str):
            try:
                best_params["hidden_layer_sizes"] = ast.literal_eval(hidden_sizes)
            except (ValueError, SyntaxError):
                pass
        return best_params, study.best_value, duration, study

    def tune_and_fit_model(self, model_name: str):
        model = self._instantiate_model(
            model_name, random_state=self.random_state, n_jobs=self.n_jobs
        )
        params, score, duration, study = self.optimize_model(model)
        fitted_model = clone(model).set_params(**params)
        preprocessor = make_preprocessor(self.preprocessor_factory, model_name)
        if preprocessor is None:
            X_train = self._as_model_input(self.X_train_raw)
            y_train = np.asarray(self.y_train_raw)
        else:
            X_train, y_train = preprocessor.fit_transform(
                self.X_train_raw.copy(), self.y_train_raw.copy()
            )
        fit_estimator(fitted_model, X_train, np.asarray(y_train))
        return fitted_model, preprocessor, score, duration, study

    def nested_cv_selection_scores(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series],
        cv_splits: int,
        cv_repeats: int,
        confidence_level: float,
    ) -> pd.DataFrame:
        rows = {model_name: [] for model_name in self.model_names}
        durations = {model_name: 0.0 for model_name in self.model_names}
        y = pd.Series(y).reset_index(drop=True)
        X = pd.DataFrame(X).reset_index(drop=True)
        groups = (
            pd.Series(groups).reset_index(drop=True) if groups is not None else None
        )
        for repeat in range(cv_repeats):
            splitter = self._make_splitter(
                y, groups=groups, requested_splits=cv_splits, repeat=repeat
            )
            for train_idx, validation_idx in splitter.split(X, y, groups):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_validation = X.iloc[validation_idx]
                y_validation = y.iloc[validation_idx]
                groups_train = groups.iloc[train_idx] if groups is not None else None
                for model_name in self.model_names:
                    start_time = time.time()
                    selector = ModelSelector(
                        X_train.to_numpy(),
                        y_train.to_numpy(),
                        X_validation.to_numpy(),
                        y_validation.to_numpy(),
                        score_metric=self.base_score_metric,
                        X_train_raw=X_train,
                        y_train_raw=y_train,
                        X_validation_raw=X_validation,
                        y_validation_raw=y_validation,
                        groups_train_raw=groups_train,
                        preprocessor_factory=self.preprocessor_factory,
                        include_models=[model_name],
                        search_profile=self.search_profile,
                        optimization_method=self.optimization_method,
                        n_iterations=self.n_iterations,
                        random_state=(
                            None
                            if self.random_state is None
                            else self.random_state + repeat
                        ),
                        n_jobs=self.n_jobs,
                        verbose=self.verbose,
                    )
                    result = selector.compare_models()
                    validation_summary = result[-2]
                    rows[model_name].append(
                        float(validation_summary.iloc[0][self.score_metric_name])
                    )
                    durations[model_name] += time.time() - start_time

        summaries = []
        for model_name, scores in rows.items():
            summaries.append(
                {
                    "model": model_name,
                    "metric": self.score_metric_name,
                    **_summarize_scores(scores, confidence_level),
                    "selection_duration": durations[model_name],
                    "status": "ok",
                }
            )
        return pd.DataFrame(summaries).sort_values(
            by=["mean_score", "std_score", "selection_duration"],
            ascending=[False, True, True],
            na_position="last",
        )

    def compare_models(self):
        best_model = None
        score_for_best_model = -np.inf
        params_for_best_model = None
        fitted_models = {}
        fitted_preprocessors = {}
        fitted_training_data = {}
        fitted_validation_data = {}
        validation_summary = pd.DataFrame()
        studies = {}

        for model in self.models:
            log.info("Optimizing model: %s", model.__class__.__name__)
            model_name = model.__class__.__name__
            params, score, duration, study = self.optimize_model(model)
            log.info(
                "Best parameters for %s: %s, score: %.4f %s",
                model_name,
                params,
                score,
                self.score_metric_name,
            )

            model = clone(model).set_params(**params)
            (
                X_train,
                y_train,
                X_validation,
                y_validation,
                preprocessor,
            ) = self._prepare_final_model_data(model_name)
            fit_estimator(model, X_train, y_train)
            fitted_models[model_name] = model
            fitted_preprocessors[model_name] = preprocessor
            fitted_training_data[model_name] = (X_train, y_train)
            fitted_validation_data[model_name] = (X_validation, y_validation)
            studies[model_name] = study

            if self.roc:
                if self.binary:
                    score_on_validation = self.score_metric(
                        y_validation,
                        model.predict_proba(X_validation)[:, 1],
                    )
                else:
                    score_on_validation = self.score_metric(
                        y_validation,
                        model.predict_proba(X_validation),
                    )
            else:
                score_on_validation = self.score_metric(
                    y_validation, model.predict(X_validation)
                )

            if score_on_validation > score_for_best_model:
                score_for_best_model = score_on_validation
                best_model = model
                params_for_best_model = params

            scores_on_validation = self._score_model_with_metrics(
                model, X_validation, y_validation
            )

            validation_summary = pd.concat(
                [
                    validation_summary,
                    pd.DataFrame(
                        [
                            {
                                "model": model_name,
                                **scores_on_validation,
                                "duration": duration,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        log.info(
            "Found best model: %s with parameters %s and score %.4f %s.",
            best_model.__class__.__name__,
            params_for_best_model,
            score_for_best_model,
            self.score_metric_name,
        )

        return (
            best_model,
            params_for_best_model,
            score_for_best_model,
            fitted_models,
            fitted_preprocessors,
            fitted_training_data,
            fitted_validation_data,
            validation_summary,
            studies,
        )

    def _prepare_final_model_data(self, model_name: str):
        if any(
            [
                self.X_train_raw is None,
                self.X_validation_raw is None,
                self.preprocessor_factory is None,
            ]
        ):
            return (
                self.X_train,
                self.y_train,
                self.X_validation,
                self.y_validation,
                None,
            )

        preprocessor = make_preprocessor(self.preprocessor_factory, model_name)
        y_train_raw = self.y_train_raw.copy()
        y_validation_raw = self.y_validation_raw.to_numpy()
        if preprocessor is None:
            return (
                self._as_model_input(self.X_train_raw),
                y_train_raw.to_numpy(),
                self._as_model_input(self.X_validation_raw),
                y_validation_raw,
                None,
            )

        X_train, y_train = preprocessor.fit_transform(
            self.X_train_raw.copy(), y_train_raw.copy()
        )
        X_validation = preprocessor.transform(self.X_validation_raw.copy())
        return (
            X_train,
            np.asarray(y_train),
            X_validation,
            y_validation_raw,
            preprocessor,
        )

    def _score_model_with_metrics(
        self, fitted_model, X_validation=None, y_validation=None
    ):
        X_validation = self.X_validation if X_validation is None else X_validation
        y_validation = (
            self.y_validation if y_validation is None else np.asarray(y_validation)
        )
        if not hasattr(fitted_model, "predict"):
            raise ValueError(
                "The model is not fitted and can not be scored with any metric."
            )

        y_pred = fitted_model.predict(X_validation)
        y_pred_proba = fitted_model.predict_proba(X_validation)
        if self.binary:
            y_pred_proba = y_pred_proba[:, 1]

        results = {
            "accuracy_score": accuracy_score(y_validation, y_pred),
            "balanced_accuracy_score": balanced_accuracy_score(y_validation, y_pred),
            "precision_score": precision_score(
                y_validation, y_pred, average="weighted", zero_division=0
            ),
            "recall_score": recall_score(
                y_validation, y_pred, average="weighted", zero_division=0
            ),
            "f1_score": f1_score(
                y_validation, y_pred, average="weighted", zero_division=0
            ),
            "jaccard_score": jaccard_score(
                y_validation, y_pred, average="weighted", zero_division=0
            ),
            "roc_auc_score": roc_auc_score(
                y_validation,
                y_pred_proba,
                multi_class="ovr",
                average="weighted",
            ),
        }

        results = {
            self.score_metric_name: results.pop(self.score_metric_name),
            **results,
        }
        return results

    @staticmethod
    def _as_model_input(data):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        return data


def repeated_cv_selection_scores(
    *,
    fitted_models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    metric_name: str,
    binary: bool,
    score_metric: Callable,
    preprocessor_factory: Optional[Callable],
    cv_splits: int,
    cv_repeats: int,
    confidence_level: float,
    random_state: Optional[int],
) -> pd.DataFrame:
    y = pd.Series(y)
    effective_splits = ModelSelector._effective_cv_splits(y, cv_splits)
    rows = []

    for model_name, estimator in fitted_models.items():
        scores = []
        start_time = time.time()
        status = "ok"
        for repeat in range(cv_repeats):
            splitter = StratifiedKFold(
                n_splits=effective_splits,
                shuffle=True,
                random_state=None if random_state is None else random_state + repeat,
            )
            for train_idx, val_idx in splitter.split(X, y):
                try:
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold = (
                        _prepare_raw_cv_fold(
                            X=X,
                            y=y,
                            model_name=model_name,
                            train_idx=train_idx,
                            val_idx=val_idx,
                            preprocessor_factory=preprocessor_factory,
                        )
                    )
                    model = clone(estimator)
                    fit_estimator(model, X_train_fold, y_train_fold)
                    if metric_name == "roc_auc_score":
                        predictions = model.predict_proba(X_val_fold)
                        if binary:
                            predictions = predictions[:, 1]
                    else:
                        predictions = model.predict(X_val_fold)
                    scores.append(
                        _compute_metric(
                            score_metric, metric_name, binary, y_val_fold, predictions
                        )
                    )
                except (
                    Exception
                ) as exc:  # pragma: no cover - data-dependent user edge cases
                    status = f"failed: {exc.__class__.__name__}"
                    scores = []
                    break
            if status != "ok":
                break

        score_summary = _summarize_scores(scores, confidence_level)
        rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                **score_summary,
                "selection_duration": time.time() - start_time,
                "status": status,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["mean_score", "std_score", "selection_duration"],
        ascending=[False, True, True],
        na_position="last",
    )


def select_from_repeated_cv_summary(
    summary: pd.DataFrame, practical_margin: float
) -> str:
    return repeated_cv_selection_decision(summary, practical_margin)["model"]


def repeated_cv_selection_decision(
    summary: pd.DataFrame, practical_margin: float
) -> dict:
    valid_summary = summary.loc[summary["status"].eq("ok")].dropna(
        subset=["mean_score"]
    )
    if valid_summary.empty:
        raise ValueError("No successful repeated-CV selection scores are available.")

    best_mean = float(valid_summary["mean_score"].max())
    contenders = valid_summary.loc[
        valid_summary["mean_score"] >= best_mean - practical_margin
    ].copy()
    contenders = contenders.sort_values(
        by=["std_score", "selection_duration", "model"],
        ascending=[True, True, True],
        na_position="last",
    )
    selected = contenders.iloc[0]
    status = "close_call" if len(contenders) > 1 else "confirmed"
    return {
        "model": str(selected["model"]),
        "status": status,
        "best_mean_score": best_mean,
        "n_contenders": int(len(contenders)),
    }


def _prepare_raw_cv_fold(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    train_idx,
    val_idx,
    preprocessor_factory: Optional[Callable],
):
    X_train_fold_raw = X.iloc[train_idx]
    X_val_fold_raw = X.iloc[val_idx]
    y_train_fold_raw = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx].to_numpy()

    if preprocessor_factory is None:
        return (
            ModelSelector._as_model_input(X_train_fold_raw),
            y_train_fold_raw.to_numpy(),
            ModelSelector._as_model_input(X_val_fold_raw),
            y_val_fold,
        )

    preprocessor = make_preprocessor(preprocessor_factory, model_name)
    if preprocessor is None:
        return (
            ModelSelector._as_model_input(X_train_fold_raw),
            y_train_fold_raw.to_numpy(),
            ModelSelector._as_model_input(X_val_fold_raw),
            y_val_fold,
        )

    X_train_fold, y_train_fold = preprocessor.fit_transform(
        X_train_fold_raw.copy(), y_train_fold_raw.copy()
    )
    X_val_fold = preprocessor.transform(X_val_fold_raw.copy())
    return X_train_fold, np.asarray(y_train_fold), X_val_fold, y_val_fold


def _compute_metric(score_metric, metric_name: str, binary: bool, y_true, predictions):
    return score_metric(y_true, predictions)


def _summarize_scores(scores, confidence_level: float) -> dict:
    score_array = pd.Series(scores, dtype="float64").dropna().to_numpy()
    n_scores = len(score_array)
    if n_scores == 0:
        return {
            "mean_score": np.nan,
            "std_score": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_scores": 0,
        }

    mean_score = float(np.mean(score_array))
    std_score = float(np.std(score_array, ddof=1)) if n_scores > 1 else 0.0
    if n_scores > 1:
        critical_value = stats.t.ppf((1 + confidence_level) / 2, df=n_scores - 1)
        margin = critical_value * std_score / np.sqrt(n_scores)
        ci_low = max(0.0, mean_score - margin)
        ci_high = min(1.0, mean_score + margin)
    else:
        ci_low = mean_score
        ci_high = mean_score

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_scores": n_scores,
    }
