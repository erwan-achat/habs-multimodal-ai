"""Nested grouped Ridge comparisons and paired participant-level metrics."""

import logging

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .signals import FEATURES
from .trials import KEY, TARGETS

ALPHAS = (1000.0, 100.0, 10.0, 1.0, 0.1)
LOG = logging.getLogger(__name__)
OUTER_FOLDS = 5
INNER_FOLDS = 3
BOOTSTRAP_DRAWS = 2000
SEED = 2026
SENSORS = tuple(FEATURES)
PRIMARY = "four_modalities"
BEST = "best_single_sensor"
FALLBACK_THRESHOLD = 0.5
FALLBACK = ("ecg", "gsr")
PREDICTION_COLUMNS = [
    *KEY, "cohort", "video", "flagged_outliers", "population", "target", "model",
    "outer_fold", "rating", "prediction", "alpha", "selected_sensor", "clip_seen",
]
FOLD_COLUMNS = ["population", "target", "outer_fold", "inner_fold", "participant_id", "role"]
METRIC_COLUMNS = [
    "population", "target", "subset", "kind", "name", "participant_id", "n_ids", "n_trials",
    "mae_macro", "mae_trials", "clip_seen_fraction", "gain", "ci_low", "ci_high", "direction",
]


def comparators(sensors, fusion_name):
    models = {sensor: ((sensor,), False) for sensor in sensors}
    models[fusion_name] = (sensors, False)
    if sensors == SENSORS:
        for excluded in sensors:
            name = "peripheral" if excluded == "eeg" else f"without_{excluded}"
            models[name] = (tuple(sensor for sensor in sensors if sensor != excluded), False)
    models["clip_ridge"] = ((), True)
    models["clip_plus_physiology"] = (sensors, True)
    return models


def complete_trials(manifest, sensors):
    return manifest["trial_status"].eq("ok") & all_sensor_statuses(manifest, sensors)


def all_sensor_statuses(manifest, sensors):
    return manifest[[f"{sensor}_status" for sensor in sensors]].eq("ok").all(axis=1)


def select_outer(data):
    n_groups = data.participant_id.nunique()
    best, best_score = [], (-1, -1, -1)
    for n_splits in range(min(OUTER_FOLDS, n_groups), 1, -1):
        splits = list(GroupKFold(n_splits).split(data, groups=data.participant_id))
        feasible = [target for target in TARGETS if target_splits(data, splits, target)]
        score = (int(TARGETS[0] in feasible), len(feasible), n_splits)
        if score > best_score:
            best, best_score = splits, score
    return best


def target_splits(data, outer, target):
    folds = []
    for fold, (train, test) in enumerate(outer, 1):
        train_data, test_data = data.iloc[train].dropna(subset=[target]), data.iloc[test].dropna(subset=[target])
        if test_data.empty:
            continue
        n_inner = min(INNER_FOLDS, train_data.participant_id.nunique())
        if n_inner < 2:
            return []
        folds.append((fold, train_data, test_data,
                      list(GroupKFold(n_inner).split(train_data, groups=train_data.participant_id))))
    return folds


def design(train, test, sensors, include_clip):
    columns = [
        column for sensor in sensors for column in FEATURES[sensor]
        if train[column].notna().any()
    ]
    if columns:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_numeric = scaler.fit_transform(imputer.fit_transform(train[columns]))
        test_numeric = scaler.transform(imputer.transform(test[columns]))
    if include_clip:
        encoder = OneHotEncoder(handle_unknown="ignore")
        train_clip = encoder.fit_transform(train[["video"]])
        test_clip = encoder.transform(test[["video"]])
        if columns:
            return hstack([train_clip, train_numeric], format="csr"), hstack([test_clip, test_numeric], format="csr")
        return train_clip, test_clip
    return train_numeric, test_numeric


def ridge_predict(x_train, y_train, x_test, alpha, has_clip):
    estimator = Ridge(alpha=alpha, solver="lsqr" if has_clip else "cholesky")
    return estimator.fit(x_train, y_train).predict(x_test)


def choose_alpha(train, inner, target, sensors, include_clip):
    errors = {alpha: [] for alpha in ALPHAS}
    for fit, validation in inner:
        fit_data, val_data = train.iloc[fit], train.iloc[validation]
        x_train, x_val = design(fit_data, val_data, sensors, include_clip)
        for alpha in ALPHAS:
            prediction = ridge_predict(x_train, fit_data[target], x_val, alpha, include_clip)
            absolute = np.abs(val_data[target].to_numpy() - prediction)
            errors[alpha].append(pd.Series(absolute, index=val_data.participant_id))
    scores = {alpha: pd.concat(parts).groupby(level=0).mean().mean() for alpha, parts in errors.items()}
    alpha = min(ALPHAS, key=scores.get)
    return alpha, scores[alpha]


def record_groups(rows, population, target, outer, inner, groups, role):
    rows.extend(
        {"population": population, "target": target, "outer_fold": outer, "inner_fold": inner,
         "participant_id": participant, "role": role}
        for participant in sorted(groups.unique())
    )


def evaluate_population(data, population, sensors, predictions, groups):
    outer = select_outer(data)
    for fold, (_, test) in enumerate(outer, 1):
        record_groups(groups, population, None, fold, None, data.iloc[test].participant_id, "assignment")
    fusion_name = PRIMARY if sensors == SENSORS else "ecg_gsr"
    models = comparators(sensors, fusion_name)
    summary = {"outer_folds": len(outer), "targets": {}}
    for target in TARGETS:
        folds = target_splits(data, outer, target)
        summary["targets"][target] = {
            "status": "evaluated" if folds else "insufficient_groups",
            "n_ids": data.loc[data[target].notna(), "participant_id"].nunique(),
            "outer_folds_used": len(folds),
            "pilot": bool(folds) and (len(folds) < OUTER_FOLDS or any(len(inner) < INNER_FOLDS for _, _, _, inner in folds)),
        }
        if not folds:
            LOG.info("%s / %s. Plis imbriqués indisponibles pour %d ID(s) éligible(s).",
                     population, target, summary["targets"][target]["n_ids"])
        for current, (fold, train, test, inner) in enumerate(folds, 1):
            LOG.info("%s / %s. Entraînement du pli externe %d/%d. %d essais train, %d essais test.",
                     population, target, current, len(folds), len(train), len(test))
            record_groups(groups, population, target, fold, None, train.participant_id, "train")
            record_groups(groups, population, target, fold, None, test.participant_id, "test")
            for inner_id, (fit, validation) in enumerate(inner, 1):
                record_groups(groups, population, target, fold, inner_id, train.iloc[fit].participant_id, "train")
                record_groups(groups, population, target, fold, inner_id, train.iloc[validation].participant_id, "validation")

            outputs = {}
            global_median = train[target].median()
            clip_medians = train.groupby("video")[target].median()
            outputs["median"] = (global_median, None, None)
            outputs["clip_median"] = (test.video.map(clip_medians).fillna(global_median).to_numpy(), None, None)
            validation_scores = {}
            for model, (inputs, has_clip) in models.items():
                alpha, score = choose_alpha(train, inner, target, inputs, has_clip)
                x_train, x_test = design(train, test, inputs, has_clip)
                fitted = ridge_predict(x_train, train[target], x_test, alpha, has_clip)
                outputs[model] = (fitted, alpha, model if model in sensors else None)
                validation_scores[model] = score

            winner = min(sensors, key=validation_scores.get)
            outputs[BEST] = outputs[winner]
            for model, (values, alpha, selected_sensor) in outputs.items():
                rows = test[[*KEY, "cohort", "video", "flagged_outliers"]].copy()
                rows = rows.assign(
                    population=population, target=target, model=model, outer_fold=fold,
                    rating=test[target], prediction=values, alpha=alpha, selected_sensor=selected_sensor,
                    clip_seen=test.video.isin(clip_medians.index),
                )
                predictions.append(rows)
    return summary


def evaluate(manifest, features, development_ids):
    candidate = manifest.merge(features, on=KEY, how="left", validate="one_to_one")
    candidate = candidate.loc[~candidate.participant_id.isin(development_ids)]
    predictions, groups = [], []
    populations = {}
    complete = candidate[complete_trials(candidate, SENSORS)]
    populations["four_modalities"] = evaluate_population(
        complete, "four_modalities", SENSORS, predictions, groups
    )

    # Use a separate population when four-modality coverage is low for the main target.
    arousal = TARGETS[0]
    matched = candidate[candidate.trial_status.eq("ok") & candidate[arousal].notna()]
    retained = complete.dropna(subset=[arousal])
    coverage = (
        min(len(retained) / len(matched), retained.participant_id.nunique() / matched.participant_id.nunique())
        if len(matched) else None
    )
    main_status = populations["four_modalities"]["targets"][arousal]["status"]
    if coverage is not None and (coverage < FALLBACK_THRESHOLD or main_status != "evaluated"):
        fallback = candidate[complete_trials(candidate, FALLBACK)]
        populations["ecg_gsr"] = evaluate_population(fallback, "ecg_gsr", FALLBACK, predictions, groups)
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=PREDICTION_COLUMNS),
        pd.DataFrame(groups, columns=FOLD_COLUMNS),
        populations,
    )


def paired_interval(gains):
    values = gains.to_numpy()
    if len(values) < 2:
        return np.nan, np.nan
    indices = np.random.default_rng(SEED).integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    return np.quantile(values[indices].mean(axis=1), (0.025, 0.975))


def check_trial_alignment(reference, compared, columns):
    """Reject even a single missing trial within an otherwise matching ID."""
    for rows in (reference, compared):
        if rows.duplicated(KEY).any() or rows[KEY].isna().any().any():
            raise ValueError("Duplicate or missing trial key")
    left = reference.set_index(KEY)[columns].sort_index()
    right = compared.set_index(KEY)[columns].sort_index()
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True,
                                      check_index_type=False)
    except AssertionError as exc:
        raise ValueError("Trial keys, folds, ratings or trial metadata differ between models") from exc


def check_paired_rows(reference, compared):
    for rows in (reference, compared):
        if not np.isfinite(rows[["rating", "prediction"]]).all().all():
            raise ValueError("Non-finite rating/prediction")
    check_trial_alignment(reference, compared, ["outer_fold", "rating", "flagged_outliers", "video", "clip_seen"])


def summarize(predictions, contrasts=None):
    records = []
    for (population, target), data in predictions.groupby(["population", "target"], sort=False):
        fusion = PRIMARY if population == "four_modalities" else "ecg_gsr"
        pairs = [(BEST, fusion), ("clip_ridge", "clip_plus_physiology")] if contrasts is None else contrasts
        by_model = {model: group for model, group in data.groupby("model", sort=False)}
        for reference, improved in pairs:
            check_paired_rows(by_model[reference], by_model[improved])
        for subset, selected in [("all", data), ("flagged_excluded", data[data.flagged_outliers.ne(1)])]:
            if subset != "all" and (selected.empty or len(selected) == len(data)):
                continue
            base = {"population": population, "target": target, "subset": subset}
            selected = selected.assign(absolute_error=(selected.rating - selected.prediction).abs())
            model_means = {}
            for model, rows in selected.groupby("model", sort=False):
                per_id = rows.groupby("participant_id").absolute_error.agg(["mean", "size"])
                model_means[model] = per_id["mean"]
                if subset == "all":
                    records.extend(
                        {**base, "kind": "participant", "name": model, "participant_id": identifier,
                         "n_ids": 1, "n_trials": count, "mae_macro": mae}
                        for identifier, mae, count in per_id.itertuples()
                    )
                records.append({**base, "kind": "model", "name": model, "n_ids": len(per_id),
                                "n_trials": len(rows), "mae_macro": per_id["mean"].mean(),
                                "mae_trials": rows.absolute_error.mean(),
                                "clip_seen_fraction": rows.clip_seen.mean()})
            for reference, improved in pairs:
                gains = model_means[reference] - model_means[improved]
                if contrasts is not None and subset == "all":
                    counts = selected[selected.model.eq(improved)].groupby("participant_id").size()
                    records.extend(
                        {**base, "kind": "participant_gain", "name": f"{reference}_vs_{improved}",
                         "participant_id": pid, "n_ids": 1, "n_trials": counts[pid], "gain": gain}
                        for pid, gain in gains.items()
                    )
                low, high = paired_interval(gains)
                direction = "positive" if low > 0 else "negative" if high < 0 else "uncertain"
                records.append({**base, "kind": "paired_gain", "name": f"{reference}_vs_{improved}",
                                "n_ids": len(gains), "n_trials": selected.model.eq(improved).sum(),
                                "gain": gains.mean(), "ci_low": low, "ci_high": high, "direction": direction})
    return pd.DataFrame(records, columns=METRIC_COLUMNS)
