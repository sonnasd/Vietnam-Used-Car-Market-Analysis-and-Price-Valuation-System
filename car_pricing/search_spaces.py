SEARCH_SPACES = {
    "Linear Regression": {
        "model__fit_intercept": [True, False],
    },
    "Decision Tree": {
        "model__max_depth": [8, 12, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": [None, 0.7, 1.0],
    },
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [12, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.7, 1.0],
    },
    "XGBoost": {
        "model__n_estimators": [300, 400, 600],
        "model__max_depth": [3, 4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    },
}
