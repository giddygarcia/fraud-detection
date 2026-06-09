import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def score_predictions(
    y_pred_train,
    y_pred_val,
    y_train,
    y_val,
    y_proba_train,
    y_proba_val,
    model_name,
    params,
    results=None,
):
    if results is None:
        results = pd.DataFrame(
            columns=[
                "Model",
                "PR AUC",
                "F1 Score",
                "Train F1",
                "Precision",
                "Recall",
                "Params",
            ]
        )

    row = {
        "Model": model_name,
        "PR AUC": average_precision_score(y_val, y_proba_val),
        "F1 Score": f1_score(y_val, y_pred_val, zero_division=0),
        "Train F1": f1_score(y_train, y_pred_train, zero_division=0),
        "Precision": precision_score(y_val, y_pred_val, zero_division=0),
        "Recall": recall_score(y_val, y_pred_val, zero_division=0),
        "Params": str(params),
    }

    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    metric_cols = ["PR AUC", "F1 Score", "Train F1", "Precision", "Recall"]
    new_results = results.sort_values(by="PR AUC", ascending=False).reset_index(
        drop=True
    )
    new_results[metric_cols] = new_results[metric_cols].astype(float).round(3)

    return results, new_results


def objective_lgbm(trial, preprocessor, X_train, y_train, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = LGBMClassifier(
        **params,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(period=-1),
        ],
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
    )

    return scores.mean()


def objective_xgb(trial, preprocessor, X_train, y_train, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }

    model = XGBClassifier(
        **params,
        scale_pos_weight=int((y_train == 0).sum() / (y_train == 1).sum()),
        random_state=42,
        verbosity=0,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
    )

    return scores.mean()
