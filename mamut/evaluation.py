import base64
import math
import os
import platform
import time
import warnings
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import psutil
import seaborn as sns

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="shap\\.plots\\.colors\\._colorconv",
)

import shap
from jinja2 import Environment, FileSystemLoader
from matplotlib import gridspec
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from mamut.preprocessing.handlers import handle_outliers
from mamut.utils.utils import model_param_dict, preprocessing_steps


def _get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _generate_experiment_setup_table():
    system_info = {
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "System": platform.system(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Python Version": platform.python_version(),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
        "CPU Cores": psutil.cpu_count(logical=True),
    }

    df = pd.DataFrame(system_info.items(), columns=["Attribute", "Value"])
    html_table = df.to_html(index=False)
    return html_table


def _generate_dataset_overview(
    X: pd.DataFrame, y: pd.Series
) -> (List[int], pd.DataFrame, pd.DataFrame):
    n_observations, n_features = X.shape

    n_rows_missing = X.isnull().any(axis=1).sum()

    # Calculate the number of outliers according to IsolationForest method:
    # TODO: Check if correct
    _, y_new, _ = handle_outliers(X, y, X.columns)
    n_outliers = len(y) - len(y_new)

    dataset_basic_list = [n_observations, n_features, n_rows_missing, n_outliers]

    feature_summary = X.dtypes.reset_index()
    feature_summary.columns = ["Feature", "Data Type"]
    feature_summary["Type"] = feature_summary["Data Type"].apply(
        lambda dt: (
            "Categorical"
            if dt == "object" or isinstance(dt, pd.CategoricalDtype)
            else "Numerical"
        )
    )
    if len(feature_summary) > 10:
        feature_summary = feature_summary.head(10)
    feature_summary = feature_summary[["Feature", "Type", "Data Type"]]

    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    class_distribution = y.value_counts().reset_index()
    class_distribution.columns = ["Class", "Count"]

    return dataset_basic_list, feature_summary, class_distribution


def _generate_preprocessing_steps_list(steps) -> str:
    categorized_steps = {}

    for step in steps:
        if step in preprocessing_steps:
            category, description = preprocessing_steps[step]
            if category not in categorized_steps:
                categorized_steps[category] = []
            categorized_steps[category].append(
                f"<strong>{step}</strong>: {description}"
            )

    html_prep_list = ""
    for category, tools in categorized_steps.items():
        html_prep_list += f"<li style='padding-left: 10px;'><strong>{category}</strong><ul style='list-style-type: '&#x2192'; margin-left: 20px;'>"
        for tool in tools:
            html_prep_list += f"<li>{tool}</li>"
        html_prep_list += "</ul></li>"

    return html_prep_list


def _generate_preprocessing_steps_html(report):
    """
    Generates an HTML list of preprocessing steps based on the report dictionary.

    Parameters
    ----------
    report : dict
        The report dictionary containing preprocessing steps.

    Returns
    -------
    str
        An HTML string representing the preprocessing steps.
    """
    html_list = ""

    for category, steps in report.items():
        html_list += f"<li><strong>{category.capitalize()}</strong><ul>"
        if isinstance(steps, dict):
            for step, details in steps.items():
                if isinstance(details, dict):
                    html_list += f"<li><strong>{step.capitalize()}</strong><ul>"
                    for sub_step, description in details.items():
                        html_list += f"<li>{sub_step}: {description}</li>"
                    html_list += "</ul></li>"
                else:
                    html_list += f"<li>{step}: {details}</li>"
        else:
            html_list += f"<li>{steps}</li>"
        html_list += "</ul></li>"

    return html_list


def _generate_models_list(excluded_models: List[str]) -> List[str]:
    all_models = model_param_dict.keys()
    available_models = [model for model in all_models if model not in excluded_models]

    return available_models


def _generate_ensemble_list(ensemble: Pipeline) -> str:
    if not ensemble:
        return ""
    ensemble = ensemble.named_steps["model"]
    base_estimators = ensemble.estimators
    meta = ensemble.final_estimator
    # Generate HTML list with ensemble contents:
    html_list = ""

    html_list += "<li><strong>Base Estimators:</strong><ul>"
    for name, estimator in base_estimators:
        html_list += f"<li>{name}: {estimator.__class__.__name__}</li>"
    html_list += "</ul></li>"

    html_list += f"<li><strong>Meta Model:</strong> <ul><li>{meta.__class__.__name__}</li></ul></li>"

    return html_list


class ModelEvaluator:

    report_template_path: str = os.path.join(os.path.dirname(__file__), "utils")

    def __init__(
        self,
        models: dict,
        # X_test and y_test are preprocessed. X and y are not.
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series,
        optimizer: str,
        n_trials: int,
        metric: str,
        studies: dict,
        training_summary: pd.DataFrame,
        pca_loadings,
        binary: bool,
        preprocessing_steps,
        is_ensemble: bool,
        greedy_ensemble,
        excluded_models: List[str] = None,
        n_top_models: int = 3,
    ):

        self.models = models
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.optimizer = optimizer
        self.n_trials = n_trials
        self.metric = metric
        self.studies = studies
        self.training_summary = training_summary
        self.pca_loadings = pca_loadings
        self.binary = binary
        self.is_ensemble = is_ensemble
        self.greedy_ensemble = greedy_ensemble
        if self.pca_loadings is not None:
            self.pca = True
        else:
            self.pca = False
        if self.training_summary is None:
            raise ValueError(
                "You need to .fit() your models before evaluating them with .evaluate()"
            )
        self.preprocessing_steps = preprocessing_steps
        self.excluded_models = excluded_models if excluded_models else []

        self.report_output_path = os.path.join(os.getcwd(), "mamut_report")
        self.plot_output_path = os.path.join(self.report_output_path, "plots")

        self.n_top_models = n_top_models

        # Create the report directory it doesn't exist:
        os.makedirs(self.report_output_path, exist_ok=True)
        os.makedirs(self.plot_output_path, exist_ok=True)
        self._set_plt_style()

    def _set_plt_style(self) -> None:
        sns.set_context("notebook", font_scale=1.05)
        plt.style.use("fivethirtyeight")
        # Set background color of all plots to #f0f8ff;
        plt.rcParams["axes.facecolor"] = "#f0f8ff"
        plt.rcParams["figure.facecolor"] = "#f0f8ff"

    def evaluate(self, training_summary: pd.DataFrame):
        return self.evaluate_to_html(training_summary)

    def plot_results_in_notebook(self):
        if self.binary:
            self._plot_roc_auc_curve(
                training_summary=self.training_summary,
                n_top=self.n_top_models,
                show=True,
                save=False,
            )
        else:
            self._plot_roc_auc_curve_multiclass(
                training_summary=self.training_summary,
                n_top=self.n_top_models,
                show=True,
                save=False,
            )
        self._plot_confusion_matrices(
            n_top=self.n_top_models,
            show=True,
            save=False,
            training_summary=self.training_summary,
        )
        self._plot_hyperparameter_tuning_history(
            n_top=self.n_top_models,
            show=True,
            save=False,
            training_summary=self.training_summary,
        )
        return

    def _plot_roc_auc_curve(
        self,
        training_summary: pd.DataFrame,
        n_top: int = 3,
        show: bool = False,
        save: bool = True,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_models = training_summary["Model"].head(n_top).to_numpy()

        for model_name in top_models:
            model = next(
                m for m in self.models.values() if m.__class__.__name__ == model_name
            )
            y_pred = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred)
            ax.plot(fpr, tpr, lw=1.5, label=f"{model_name} ROC ({auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plot_output_path, "roc_auc_curve.png"),
                format="png",
                bbox_inches="tight",
            )
        if show:
            plt.show()

        plt.close(fig)

        return

    def _plot_roc_auc_curve_multiclass(
        self,
        training_summary: pd.DataFrame,
        n_top: int = 3,
        show: bool = False,
        save: bool = True,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_models = training_summary["Model"].head(n_top).to_numpy()
        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))

        for model_name in top_models:
            model = next(
                m for m in self.models.values() if m.__class__.__name__ == model_name
            )
            y_score = model.fit(self.X_train, self.y_train).predict_proba(self.X_test)
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                lw=2,
                label=f"Micro-averaged {model_name} (area = {roc_auc:0.2f})",
            )

        ax.plot([0, 1], [0, 1], "k--", lw=2)
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_title("Micro-Averaged ROC Curve (One-vs-Rest)", fontsize=14)
        ax.legend(loc="lower right", fontsize=12)
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plot_output_path, "roc_auc_curve.png"),
                format="png",
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close(fig)

        return

    def _plot_confusion_matrices(
        self,
        training_summary: pd.DataFrame,
        n_top: int = 3,
        show: bool = False,
        save: bool = True,
    ) -> None:
        rows = math.ceil(n_top / 3)
        fig = plt.figure(figsize=(18, 5 * rows))
        top_models = training_summary["Model"].head(n_top).to_numpy()
        if n_top == 3:
            gs = gridspec.GridSpec(1, 3, wspace=0.4)
        elif n_top > 3:
            gs = gridspec.GridSpec(rows, 3, wspace=0.3, hspace=0.3)
        else:
            gs = gridspec.GridSpec(1, n_top, wspace=0.3, hspace=0.3)

        for i, model_name in enumerate(top_models):
            model = next(
                m for m in self.models.values() if m.__class__.__name__ == model_name
            )
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            ax = fig.add_subplot(gs[i])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            plt.title(f"{model_name}", fontsize=14)
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("Actual", fontsize=12)

        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plot_output_path, "confusion_matrices.png"),
                format="png",
                bbox_inches="tight",
            )
        if show:
            plt.show()

        plt.close()

        return

    def _plot_hyperparameter_tuning_history(
        self,
        training_summary: pd.DataFrame,
        n_top: int = 3,
        show: bool = False,
        save: bool = True,
    ) -> None:
        self._set_plt_style()
        top_models = training_summary["Model"].head(n_top).to_numpy()

        for i, model_name in enumerate(top_models):
            study = self.studies.get(model_name)
            if study:
                plt.figure(figsize=(6, 5), facecolor="#f0f8ff")
                ax = optuna.visualization.matplotlib.plot_optimization_history(study)
                if not show:
                    ax.set_facecolor("#f0f8ff")
                    ax.spines["top"].set_color("#007bb5")
                    ax.spines["right"].set_color("#007bb5")
                    ax.spines["bottom"].set_color("#007bb5")
                    ax.spines["left"].set_color("#007bb5")
                    ax.grid(color="grey")  # Change grid color to grey
                ax.legend().set_visible(False)  # Remove legend
                plt.title(f"{model_name} Tuning History", fontsize=14)
                plt.xlabel("Trial", fontsize=12)
                plt.ylabel(f"{self.metric} Value", fontsize=12)
                plt.tight_layout()

                if save:
                    plt.savefig(
                        os.path.join(
                            self.plot_output_path,
                            f"hyperparameter_tuning_history_{i + 1}.png",
                        ),
                        format="png",
                        bbox_inches="tight",
                    )
                if show:
                    plt.show()

                plt.close()

        return

    def _plot_feature_importances(self, show: bool = False, save: bool = True) -> None:
        self._set_plt_style()
        rf = ExtraTreesClassifier(random_state=42)
        rf.fit(self.X_train, self.y_train)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        if len(indices) > 10:
            indices = indices[:10]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), self.X.columns[indices], rotation=90)
        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Importance", fontsize=12)
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plot_output_path, "feature_importance.png"),
                format="png",
                bbox_inches="tight",
            )
        if show:
            plt.show()

        plt.close()

        return

    def _plot_shap_beeswarm(self, model, show: bool = False, save: bool = True) -> None:
        if model.__class__.__name__ in ["KNeighborsClassifier", "SVC", "MLPClassifier"]:
            explainer = shap.Explainer(model.predict, self.X_train)
        else:
            explainer = shap.Explainer(model, self.X_train)
        shap_values = explainer(self.X_train)

        if len(shap_values.shape) == 3:
            num_classes = shap_values.shape[2]
            for class_idx in range(num_classes):
                plt.figure(figsize=(10, 6))
                shap.plots.beeswarm(shap_values[:, :, class_idx], show=False)
                plt.title(f"SHAP Beeswarm Plot For Class {class_idx}", fontsize=14)
                plt.tight_layout()

                if save:
                    plt.savefig(
                        os.path.join(
                            self.plot_output_path,
                            f"shap_beeswarm_class_{class_idx}.png",
                        ),
                        format="png",
                        bbox_inches="tight",
                    )
                if show:
                    plt.show()

                plt.close()
        else:
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            plt.title("SHAP Beeswarm Plot", fontsize=14)
            plt.tight_layout()

            if save:
                plt.savefig(
                    os.path.join(self.plot_output_path, "shap_values.png"),
                    format="png",
                    bbox_inches="tight",
                )
            if show:
                plt.show()
            plt.close()
        return

    def _plot_pca_loadings(self, show: bool = False, save: bool = True) -> None:
        if self.pca_loadings is None:
            raise ValueError(
                "PCA loadings are not available. "
                "Potentially PCA was not used in the preprocessing steps."
                "Use Mamut(pca=True) to include PCA in the preprocessing steps."
            )

        self._set_plt_style()
        sns.set_palette(sns.color_palette("tab20", 20))

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.pca_loadings,
            annot=False,
            cmap="coolwarm",
            xticklabels=self.X.columns,
            yticklabels=[f"PC{i + 1}" for i in range(self.pca_loadings.shape[0])],
        )
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Principal Components", fontsize=12)
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plot_output_path, "pca_loadings_heatmap.png"),
                format="png",
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close()

        return

    def _plot_pca_loadings2(self, show: bool = False, save: bool = True) -> None:
        if self.pca_loadings is None:
            raise ValueError(
                "PCA loadings are not available. "
                "Potentially PCA was not used in the preprocessing steps."
                "Use Mamut(pca=True) to include PCA in the preprocessing steps."
            )
        self._set_plt_style()
        sns.set_palette(sns.color_palette("tab20", 20))
        n_components = self.pca_loadings.shape[0]
        n_features = self.pca_loadings.shape[1]

        plt.figure(figsize=(10, 6))
        for i in range(n_components):
            plt.bar(
                np.arange(n_features) + i / n_components,
                self.pca_loadings[i],
                width=1 / n_components,
                label=f"PC{i + 1}",
            )

        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Loadings", fontsize=12)
        plt.title("PCA Loadings", fontsize=14)
        plt.xticks(np.arange(n_features), self.X.columns, rotation=90)
        plt.legend(loc="best")
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plot_output_path, "pca_loadings.png"),
                format="png",
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close()

        return

    def _generate_greedy_ensemble_results_html(self, greedy_ensemble):
        """
        Generates an HTML table with the results of the greedy ensemble.

        Parameters
        ----------
        greedy_ensemble : sklearn.pipeline.Pipeline
            The greedy ensemble pipeline containing preprocessing steps and stacking classifier.

        Returns
        -------
        str
            An HTML string representing the results of the greedy ensemble.
        """
        if not greedy_ensemble:
            return ""
        greedy_ensemble = greedy_ensemble.named_steps["model"]
        results = self._score_model_with_metrics(greedy_ensemble)

        results_df = pd.DataFrame(
            [
                {
                    "model": "Greedy Ensemble",
                    **results,
                }
            ]
        )
        results_df = results_df.rename(
            columns={
                "model": "Model",
                "accuracy_score": "Accuracy",
                "balanced_accuracy_score": "Balanced Accuracy",
                "precision_score": "Precision",
                "recall_score": "Recall",
                "f1_score": "F1 Score",
                "jaccard_score": "Jaccard Score",
                "roc_auc_score": "ROC AUC",
            }
        )

        html_table = results_df.to_html(index=False)
        return html_table

    def evaluate_to_html(
        self,
        training_summary: pd.DataFrame,
    ):
        # Check if the training_summary is a DataFrame and not empty!:
        if (
            training_summary is None
            or not isinstance(training_summary, pd.DataFrame)  # noqa
            or training_summary.empty  # noqa
        ):
            raise ValueError(
                "Can't produce a HTML report because training_summary should be a DataFrame and not empty."
            )

        training_summary = training_summary.rename(
            columns={
                "model": "Model",
                "accuracy_score": "Accuracy",
                "balanced_accuracy_score": "Balanced Accuracy",
                "precision_score": "Precision",
                "recall_score": "Recall",
                "f1_score": "F1 Score",
                "jaccard_score": "Jaccard Score",
                "roc_auc_score": "ROC AUC",
                "duration": "Training Time [s]",
            }
        )
        # Sort the training_summary DataFrame by the score_metric column
        training_summary = training_summary.sort_values(
            by=training_summary.columns[1], ascending=False
        ).reset_index(drop=True)

        self.training_summary = training_summary

        styled_training_summary = training_summary.style.apply(
            _highlight_first_cell, axis=1
        )

        # Transform summary to HTML:
        training_summary_html = styled_training_summary.to_html()
        image_header_path = os.path.join(self.report_template_path, "mamut_header.png")
        base64_image = _get_base64_image(image_header_path)

        dataset_basic_list, feature_summary, class_distribution = (
            _generate_dataset_overview(self.X, self.y)
        )

        if self.binary:
            self._plot_roc_auc_curve(training_summary)
        else:
            self._plot_roc_auc_curve_multiclass(training_summary)

        self._plot_confusion_matrices(training_summary)
        self._plot_hyperparameter_tuning_history(training_summary)
        self._plot_feature_importances()
        best_model_name = training_summary.iloc[0]["Model"]
        best_model = self.models[best_model_name]

        self._plot_shap_beeswarm(best_model)

        if self.pca:
            self._plot_pca_loadings()

        # Load the Jinja2 template placed in report_template_path:
        env = Environment(loader=FileSystemLoader(self.report_template_path))
        template = env.get_template("report_template.html")

        # Render the template with the training_summary and save the HTML file
        time_signature = str(time.strftime(" %d %B %Y, %I:%M %p", time.localtime()))

        html_content = template.render(
            time_signature=time_signature,
            training_summary=training_summary_html,
            image_header=base64_image,
            experiment_setup=_generate_experiment_setup_table(),
            models_evaluated=_generate_models_list(self.excluded_models),
            optimizer=(
                "Tree-structured Parzen Estimator"
                if self.optimizer == "bayes"
                else "Random Search"
            ),
            metric=self.metric,
            n_trials=self.n_trials,
            best_model=training_summary.iloc[0]["Model"],
            basic_dataset_info=dataset_basic_list,
            feature_summary=feature_summary.to_html(index=False),
            class_distribution=class_distribution.to_html(index=False),
            feature_importance_method="Extra Trees Importances",
            pca=self.pca,
            binary=self.binary,
            is_ensemble=self.is_ensemble,
            ensemble_method="Stacking",
            ensemble_list=_generate_ensemble_list(self.greedy_ensemble),
            ensemble_summary=self._generate_greedy_ensemble_results_html(
                self.greedy_ensemble
            ),
            preprocessing_list=_generate_preprocessing_steps_html(
                self.preprocessing_steps
            ),
        )

        time_signature = datetime.strptime(
            time_signature.strip(), "%d %B %Y, %I:%M %p"
        ).strftime("%d-%m-%Y_%H-%M")
        with open(
            os.path.join(self.report_output_path, f"report_{time_signature}.html"), "w"
        ) as f:
            f.write(html_content)

        return html_content

    def _score_model_with_metrics(self, fitted_model):
        if not hasattr(fitted_model, "predict"):
            raise ValueError(
                "The model is not fitted and can not be scored with any metric."
            )

        y_pred = fitted_model.predict(self.X_test)
        y_pred_proba = fitted_model.predict_proba(self.X_test)
        if self.binary:
            y_pred_proba = y_pred_proba[:, 1]

        results = {
            "accuracy_score": accuracy_score(self.y_test, y_pred),
            "balanced_accuracy_score": balanced_accuracy_score(self.y_test, y_pred),
            "precision_score": precision_score(self.y_test, y_pred, average="weighted"),
            "recall_score": recall_score(self.y_test, y_pred, average="weighted"),
            "f1_score": f1_score(self.y_test, y_pred, average="weighted"),
            "jaccard_score": jaccard_score(self.y_test, y_pred, average="weighted"),
            "roc_auc_score": roc_auc_score(
                self.y_test, y_pred_proba, multi_class="ovr", average="weighted"
            ),
        }

        results = {
            self.metric: results.pop(self.metric),
            **results,
        }
        return results


def _highlight_first_cell(s):
    return [
        (
            "background-color: yellow"
            if (i == 0 and s.name == 0) or (i == 1 and s.name == 0)
            else ""
        )
        for i in range(len(s))
    ]
