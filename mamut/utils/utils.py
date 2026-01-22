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

lr_params = {
    "C": (1e-4, 1e4, "log"),
    "l1_ratio": (1e-4, 1.0, "log"),
    "class_weight": (["balanced"], "categorical"),
    "max_iter": (1000, 1000, "int"),
    "solver": (["saga", "lbfgs", "liblinear"], "categorical"),
}

tree_params = {
    "n_estimators": (10, 1000, "int"),
    "criterion": (["gini", "entropy", "log_loss"], "categorical"),
    "bootstrap": ([True], "categorical"),
    "max_samples": (0.5, 1, "float"),
    "max_features": (0.1, 0.9, "float"),
    "min_samples_leaf": (0.05, 0.25, "float"),
}

xgb_params = {
    "n_estimators": (10, 1000, "int"),
    "learning_rate": (1e-4, 0.4, "log"),
    "subsample": (0.25, 1.0, "float"),
    "booster": (["gbtree"], "categorical"),
    "max_depth": (1, 15, "int"),
    "min_child_weight": (1, 128, "float"),
    "colsample_bytree": (0.2, 1.0, "float"),
    "colsample_bylevel": (0.2, 1.0, "float"),
    "reg_alpha": (1e-4, 512.0, "log"),
    "reg_lambda": (1e-3, 1e3, "log"),
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

model_param_dict = {
    "LogisticRegression": lr_params,
    "RandomForestClassifier": tree_params,
    "SVC": svc_params,
    "XGBClassifier": xgb_params,
    "MLPClassifier": mlp_params,
    "GaussianNB": gnb_params,
    "KNeighborsClassifier": knn_params,
}

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
            param_dict["l1_ratio"] = None

    return param_dict
