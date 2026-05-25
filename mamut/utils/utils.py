from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEARCH_PROFILES = ("quick", "balanced", "thorough")

lr_params = {
    "C": (1e-4, 1e4, "log"),
    "l1_ratio": (1e-4, 1.0, "log"),
    "class_weight": ([None, "balanced"], "categorical"),
    "max_iter": (1000, 1000, "int"),
    "solver": (["saga", "lbfgs", "liblinear"], "categorical"),
}

tree_params = {
    "n_estimators": (100, 500, "int"),
    "criterion": (["gini", "entropy", "log_loss"], "categorical"),
    "bootstrap": ([True], "categorical"),
    "max_depth": ([None, 5, 10, 20, 30], "categorical"),
    "max_features": (["sqrt", "log2", None], "categorical"),
    "min_samples_leaf": (1, 10, "int"),
    "min_samples_split": (2, 20, "int"),
    "class_weight": ([None, "balanced_subsample"], "categorical"),
}

extra_trees_params = {
    "n_estimators": (100, 500, "int"),
    "criterion": (["gini", "entropy", "log_loss"], "categorical"),
    "max_depth": ([None, 5, 10, 20, 30], "categorical"),
    "max_features": (["sqrt", "log2", None], "categorical"),
    "min_samples_leaf": (1, 10, "int"),
    "min_samples_split": (2, 20, "int"),
    "class_weight": ([None, "balanced"], "categorical"),
}

hist_gradient_boosting_params = {
    "max_iter": (50, 250, "int"),
    "learning_rate": (0.01, 0.2, "log"),
    "max_leaf_nodes": (15, 63, "int"),
    "max_depth": ([None, 3, 5, 8, 12], "categorical"),
    "min_samples_leaf": (10, 50, "int"),
    "l2_regularization": (1e-6, 10.0, "log"),
    "class_weight": ([None, "balanced"], "categorical"),
}

xgb_params = {
    "n_estimators": (50, 500, "int"),
    "learning_rate": (0.01, 0.3, "log"),
    "subsample": (0.5, 1.0, "float"),
    "booster": (["gbtree"], "categorical"),
    "max_depth": (2, 8, "int"),
    "min_child_weight": (1, 20, "float"),
    "colsample_bytree": (0.5, 1.0, "float"),
    "colsample_bylevel": (0.5, 1.0, "float"),
    "reg_alpha": (1e-6, 10.0, "log"),
    "reg_lambda": (1e-3, 100.0, "log"),
}

lightgbm_params = {
    "n_estimators": (50, 500, "int"),
    "learning_rate": (0.01, 0.3, "log"),
    "num_leaves": (15, 127, "int"),
    "max_depth": ([-1, 3, 5, 8, 12], "categorical"),
    "min_child_samples": (5, 60, "int"),
    "subsample": (0.5, 1.0, "float"),
    "colsample_bytree": (0.5, 1.0, "float"),
    "reg_alpha": (1e-6, 10.0, "log"),
    "reg_lambda": (1e-3, 100.0, "log"),
    "class_weight": ([None, "balanced"], "categorical"),
}

catboost_params = {
    "iterations": (50, 500, "int"),
    "learning_rate": (0.01, 0.3, "log"),
    "depth": (3, 8, "int"),
    "l2_leaf_reg": (1e-2, 20.0, "log"),
    "random_strength": (1e-3, 10.0, "log"),
    "border_count": (32, 255, "int"),
}

svc_params = {
    "C": (1e-4, 1e4, "log"),
    "kernel": (["linear", "poly", "rbf", "sigmoid"], "categorical"),
    "gamma": (1e-4, 1.0, "log"),
    "class_weight": (["balanced"], "categorical"),
    "probability": ([True], "categorical"),
}

mlp_params = {
    "hidden_layer_sizes": (
        [(32,), (64,), (128,), (256,), (32, 16), (32, 32), (64, 32), (64, 64)],
        "categorical",
    ),
    "activation": (["identity", "logistic", "tanh", "relu"], "categorical"),
    "solver": (["lbfgs", "sgd", "adam"], "categorical"),
    "alpha": (1e-5, 1e-2, "log"),
    "learning_rate": (["constant", "invscaling", "adaptive"], "categorical"),
    "learning_rate_init": (1e-4, 1e-1, "log"),
    "power_t": (0.1, 0.9, "float"),
    "max_iter": (100, 200, "int"),
    "momentum": (0.5, 0.9, "float"),
}

gnb_params = {
    "var_smoothing": (1e-9, 1e-5, "log"),
}

knn_params = {
    "n_neighbors": (1, 30, "int"),
}

MODEL_SEARCH_SPACES = {
    "LogisticRegression": lr_params,
    "RandomForestClassifier": tree_params,
    "ExtraTreesClassifier": extra_trees_params,
    "HistGradientBoostingClassifier": hist_gradient_boosting_params,
    "SVC": svc_params,
    "XGBClassifier": xgb_params,
    "LGBMClassifier": lightgbm_params,
    "CatBoostClassifier": catboost_params,
    "MLPClassifier": mlp_params,
    "GaussianNB": gnb_params,
    "KNeighborsClassifier": knn_params,
}

QUICK_MODEL_NAMES = (
    "LogisticRegression",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GaussianNB",
)
BALANCED_MODEL_NAMES = (
    "LogisticRegression",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "HistGradientBoostingClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
    "GaussianNB",
)
THOROUGH_MODEL_NAMES = tuple(MODEL_SEARCH_SPACES)

# Backward-compatible alias used by older tests and users for introspection.
model_param_dict = MODEL_SEARCH_SPACES


def model_names_for_profile(profile: str) -> tuple[str, ...]:
    if profile == "quick":
        return QUICK_MODEL_NAMES
    if profile == "balanced":
        return BALANCED_MODEL_NAMES
    if profile == "thorough":
        return THOROUGH_MODEL_NAMES
    raise ValueError("search_profile must be one of: quick, balanced, thorough.")


metric_dict = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "balanced_accuracy": balanced_accuracy_score,
    "jaccard": jaccard_score,
    "roc_auc_score": roc_auc_score,
}

preprocessing_steps = {
    "IsolationForest": (
        "Outlier detection",
        "Detects and removes outliers from the dataset.",
    ),
    "SimpleImputer": ("Imputation", "Fills missing values in the dataset."),
    "PowerTransformer": (
        "Transformation",
        "Applies power transformation to reduce skewness.",
    ),
    "OneHotEncoder": (
        "Encoding",
        "Encodes categorical features using one-hot encoding.",
    ),
    "SelectFromModel": (
        "Feature selection",
        "Selects features based on importance weights.",
    ),
    "PCA": (
        "Dimensionality reduction",
        "Reduces dimensionality of the dataset using Principal Component Analysis.",
    ),
    "ExtraTreesClassifier": (
        "Feature selection",
        "Classifier used for feature selection.",
    ),
    "StandardScaler": (
        "Scaling",
        "Standardizes features by removing the mean and scaling to unit variance.",
    ),
    "RobustScaler": (
        "Scaling",
        "Scales features using statistics that are robust to outliers.",
    ),
    "KNNImputer": ("Imputation", "Fills missing values using k-Nearest Neighbors."),
    "IterativeImputer": (
        "Imputation",
        "Fills missing values using iterative imputation.",
    ),
}


def sample_parameter(trial, param_name, value):
    """Sample a parameter value based on its distribution type."""
    if len(value) == 3:
        low, high, dist_type = value
        if dist_type == "log":
            return trial.suggest_float(param_name, low, high, log=True)
        elif dist_type == "float":
            return trial.suggest_float(param_name, low, high)
        else:
            return trial.suggest_int(param_name, low, high)
    elif len(value) == 2:
        options, dist_type = value
        if any(isinstance(option, tuple) for option in options):
            option_map = {repr(option): option for option in options}
            choice = trial.suggest_categorical(param_name, list(option_map.keys()))
            return option_map[choice]
        return trial.suggest_categorical(param_name, options)
    else:
        raise ValueError("Invalid hyperparameter search space.")


def adjust_search_spaces(param_dict, model):
    if isinstance(model, LogisticRegression):
        if param_dict["solver"] == "saga":
            param_dict["penalty"] = "elasticnet"
        else:
            param_dict["penalty"] = "l2"
            param_dict.pop("l1_ratio", None)

    return param_dict
