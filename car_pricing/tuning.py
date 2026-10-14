import json
from math import prod

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from .search_spaces import SEARCH_SPACES

SCORING = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
}


def rank_models_by_cv(model_pipelines, X_train, y_train, cv):
    rows = []
    for name, pipeline in model_pipelines.items():
        scores = cross_validate(
            clone(pipeline),
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        row = {
            "Model": name,
            "R2_cv": scores["test_R2"].mean(),
            "R2_cv_std": scores["test_R2"].std(),
            "MAE_cv": -scores["test_MAE"].mean(),
            "RMSE_cv": -scores["test_RMSE"].mean(),
        }
        for metric in SCORING:
            sign = 1 if metric == "R2" else -1
            for fold, score in enumerate(scores[f"test_{metric}"], start=1):
                row[f"{metric}_fold_{fold}"] = sign * score
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["R2_cv", "RMSE_cv", "Model"], ascending=[False, True, True]
    ).reset_index(drop=True)


def search_space_table(model_names):
    return pd.DataFrame(
        [
            {
                "Model": name,
                "Parameter": parameter.removeprefix("model__"),
                "Values": json.dumps(values, ensure_ascii=False),
                "Combinations": prod(len(v) for v in SEARCH_SPACES[name].values()),
            }
            for name in model_names
            for parameter, values in SEARCH_SPACES[name].items()
        ]
    )


def trial_results(name, search):
    results = search.cv_results_
    table = pd.DataFrame(
        {
            "Model": name,
            "Trial": np.arange(1, len(results["params"]) + 1),
            "Params": [json.dumps(p, sort_keys=True) for p in results["params"]],
            "Rank_R2_cv": results["rank_test_R2"],
            "R2_cv": results["mean_test_R2"],
            "R2_cv_std": results["std_test_R2"],
            "MAE_cv": -results["mean_test_MAE"],
            "MAE_cv_std": results["std_test_MAE"],
            "RMSE_cv": -results["mean_test_RMSE"],
            "RMSE_cv_std": results["std_test_RMSE"],
            "R2_train_cv": results["mean_train_R2"],
            "MAE_train_cv": -results["mean_train_MAE"],
            "RMSE_train_cv": -results["mean_train_RMSE"],
            "Fit_time_seconds": results["mean_fit_time"],
        }
    )
    for metric in SCORING:
        sign = 1 if metric == "R2" else -1
        for fold in range(search.n_splits_):
            table[f"{metric}_fold_{fold + 1}"] = (
                sign * results[f"split{fold}_test_{metric}"]
            )
    for parameter in search.param_distributions:
        table[parameter] = [p[parameter] for p in results["params"]]
    return table


def tune_top_models(
    model_pipelines,
    cv_ranking,
    X_train,
    y_train,
    cv,
    top_n=2,
    n_iter=12,
    random_state=42,
):
    if top_n not in (1, 2) or n_iter < 1:
        raise ValueError("Chọn 1–2 mô hình và ít nhất một bộ tham số.")
    selected_names = cv_ranking.head(top_n)["Model"].tolist()
    searches, tuned_models, trials, summaries = {}, {}, [], []
    for name in selected_names:
        space = SEARCH_SPACES[name]
        search = RandomizedSearchCV(
            clone(model_pipelines[name]),
            param_distributions=space,
            n_iter=min(n_iter, prod(len(values) for values in space.values())),
            scoring=SCORING,
            refit="R2",
            cv=cv,
            random_state=random_state,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        baseline = cv_ranking.set_index("Model").loc[name]
        use_tuned = search.best_score_ > baseline["R2_cv"]
        chosen_model = (
            search.best_estimator_
            if use_tuned
            else clone(model_pipelines[name]).fit(X_train, y_train)
        )
        chosen_params = {key: chosen_model.get_params()[key] for key in space}
        chosen_score = search.best_score_ if use_tuned else baseline["R2_cv"]
        chosen_rmse = (
            -search.cv_results_["mean_test_RMSE"][search.best_index_]
            if use_tuned
            else baseline["RMSE_cv"]
        )
        chosen_mae = (
            -search.cv_results_["mean_test_MAE"][search.best_index_]
            if use_tuned
            else baseline["MAE_cv"]
        )
        chosen_std = (
            search.cv_results_["std_test_R2"][search.best_index_]
            if use_tuned
            else baseline["R2_cv_std"]
        )
        searches[name] = search
        tuned_models[name] = chosen_model
        trials.append(trial_results(name, search))
        summaries.append(
            {
                "Model": name,
                "Trials": len(search.cv_results_["params"]),
                "R2_cv_before": baseline["R2_cv"],
                "R2_cv_search": search.best_score_,
                "R2_cv_selected": chosen_score,
                "R2_cv_selected_std": chosen_std,
                "MAE_cv_before": baseline["MAE_cv"],
                "MAE_cv_selected": chosen_mae,
                "RMSE_cv_before": baseline["RMSE_cv"],
                "RMSE_cv_selected": chosen_rmse,
                "R2_cv_gain": chosen_score - baseline["R2_cv"],
                "Selected_configuration": "Tuned" if use_tuned else "Initial",
                "Best_search_params": json.dumps(search.best_params_, sort_keys=True),
                "Selected_params": json.dumps(chosen_params, sort_keys=True),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["R2_cv_selected", "RMSE_cv_selected", "Model"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    trial_table = pd.concat(trials, ignore_index=True)
    return searches, tuned_models, trial_table, summary
