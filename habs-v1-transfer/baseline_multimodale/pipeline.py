"""Run V0 and publish the trial tables, evaluations and configuration."""

import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import pandas as pd
import pyedflib

from . import evaluation, signals, trials

REPO = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def coverage_table(manifest, development_ids):
    rows = []
    cohorts = [("all", manifest), *manifest.groupby("cohort", dropna=False)]
    for cohort, data in cohorts:
        cohort = "unknown" if pd.isna(cohort) else cohort
        scope = data.loc[~data.participant_id.isin(development_ids)]
        ready = scope.trial_status.eq("ok")
        stages = {
            "manifest": data,
            "outside_development": scope,
            "metadata_event": scope[scope.n_metadata_events.gt(0)],
            "bdf_start_marker": scope[scope.n_bdf_events.gt(0)],
            "label_row": scope[scope.n_label_rows.gt(0)],
            "matched_trial": scope[ready],
            "matched_and_labelled": scope[ready & scope[list(trials.TARGETS)].notna().any(axis=1)],
        }
        for reason, failed in data.groupby("trial_status"):
            if reason != "ok":
                stages[f"trial_status:{reason}"] = failed
        for sensor in signals.FEATURES:
            stages[sensor] = scope[ready & scope[f"{sensor}_status"].eq("ok")]
        for population, sensors in [("four_modalities", evaluation.SENSORS), ("ecg_gsr", evaluation.FALLBACK)]:
            stages[population] = scope[evaluation.complete_trials(scope, sensors)]
        for stage, selected in stages.items():
            for target in ("all", *trials.TARGETS):
                eligible = selected if target == "all" else selected.dropna(subset=[target])
                rows.append({
                    "cohort": cohort, "stage": stage, "target": target,
                    "n_trials": len(eligible), "n_ids": eligible.participant_id.nunique(),
                    "n_without_trials": scope.participant_id.nunique() - eligible.participant_id.nunique()
                    if stage != "manifest" and not stage.startswith("trial_status:") else None,
                })
    return pd.DataFrame(rows)


def input_signature(root, sessions, labels_path, markers_path):
    digest = hashlib.sha256()
    for path in (labels_path, markers_path):
        if path.is_file():
            digest.update(path.read_bytes())
    for key, files in sorted(sessions.items()):
        for kind, path in sorted(files.items()):
            digest.update(str(path.relative_to(root)).encode())
            if kind == "bdf":
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
                map_path = path.with_suffix(".record_map")
                if map_path.is_file():
                    digest.update(map_path.read_bytes())
            else:
                digest.update(path.read_bytes())
    return digest.hexdigest()


def code_signature():
    digest = hashlib.sha256()
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths.append(REPO / "starting_notebooks" / "bdf_loader.py")
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_outputs(out_dir, tables, config, inspection):
    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".v0-", dir=out_dir) as staging:
        temp = Path(staging)
        for name, table in tables.items():
            table.to_csv(temp / name, index=False)
        diagnostics = {"sample_smoke.json": inspection} if inspection is not None else {}
        for name, payload in {**diagnostics, "config.json": config}.items():
            (temp / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Publish the configuration last so it describes the accompanying tables.
        for name in (*tables, *diagnostics, "config.json"):
            (temp / name).replace(out_dir / name)


def sample_inspection(root):
    result = []
    for path in sorted((root / "PREPROCESSED").glob("P*/*_task-resting-state_multimodal.bdf")):
        with pyedflib.EdfReader(str(path)) as reader:
            source = signals.Signals(reader, path.with_suffix(".record_map"))
            features, details = source.extract(0.0, signals.WINDOW_S)
            result.append({
                "file": str(path.relative_to(root)), "duration_s": source.duration,
                "bdf_annotations": len(reader.readAnnotations()[0]),
                "features_first_10s": {key: float(value) for key, value in features.items()},
                "channels_first_10s": details,
            })
    return result


def process(data_dir: Path, out_dir: Path, *, csv_dir: Path | None = None, smoke=False, development_ids=()):
    started = monotonic()
    root, out_dir = Path(data_dir), Path(out_dir)
    if not (root / "PREPROCESSED").is_dir():
        raise FileNotFoundError(f"Dossier PREPROCESSED introuvable dans {root}")
    development_ids = tuple(sorted({value.strip().upper() for value in development_ids}))
    tables_root = root if csv_dir is None else Path(csv_dir)
    labels_path, markers_path = tables_root / "labels_train.csv", tables_root / "markers.csv"
    markers_source = "dataset"
    if csv_dir is not None:
        markers_source = "csv_dir"
        for path in (labels_path, markers_path):
            if not path.is_file():
                raise FileNotFoundError(f"Fichier CSV introuvable avec --csv-dir ({path})")
    elif not markers_path.is_file():
        markers_path = REPO / "starting_notebooks" / "markers.csv"
        markers_source = "starting_notebooks/markers.csv"
    LOG.info("Lecture des annotations depuis %s", tables_root.resolve())
    pairs = trials.read_markers(markers_path)
    labels, invalid = trials.read_labels(labels_path)
    label_count = len(invalid) + sum(len(rows) for by_marker in labels.values() for rows in by_marker.values())
    LOG.info("%d lignes de notes chargées. Recherche des sessions dans %s.", label_count, root.resolve())
    sessions = trials.find_session_files(root)
    video_count = sum("bdf" in files for files in sessions.values())
    LOG.info("%d BDF vidéo repérés pour %d sessions inventoriées. Vérification et extraction des fenêtres.",
             video_count, len(sessions))
    manifest, features = trials.build_manifest(root, sessions, pairs, labels, invalid)
    recovered = int(manifest.label_day_source.eq("verified_recording").sum())
    LOG.info("Extraction terminée. %d fenêtres vérifiées et %d jours de session récupérés.", len(features), recovered)
    complete = evaluation.complete_trials(manifest, evaluation.SENSORS)
    in_evaluation = ~manifest.participant_id.isin(development_ids)
    for target in trials.TARGETS:
        manifest[f"eligible_{target}"] = complete & in_evaluation & manifest[target].notna()
    coverage = coverage_table(manifest, development_ids)
    LOG.info("Validation croisée groupée par participant.")
    predictions, folds, populations = evaluation.evaluate(manifest, features, development_ids)
    metrics = evaluation.summarize(predictions)

    input_summary = {
        "video_bdfs": video_count,
        "session_metadata": sum("metadata" in files for files in sessions.values()),
        "labels_available": labels_path.is_file(),
        "label_rows": label_count,
        "label_days_recovered": recovered,
        "manifest_rows": len(manifest),
        "matched_trials": int(manifest.trial_status.eq("ok").sum()),
    }
    arousal = populations["four_modalities"]["targets"][trials.TARGETS[0]]
    if not input_summary["video_bdfs"] or not input_summary["labels_available"]:
        status = "missing_inputs"
    else:
        status = "evaluated" if arousal["status"] == "evaluated" else "insufficient_main_population"
    config = {
        "status": status,
        "input": input_summary,
        "data_signature": input_signature(root, sessions, labels_path, markers_path),
        "data_signature_basis": "CSV, JSON et record maps par contenu, BDF par chemin relatif, taille et date",
        "code_sha256": code_signature(),
        "versions": {package: version(package) for package in ("numpy", "pandas", "pyedflib", "scipy", "scikit-learn", "mne")},
        "window_s": signals.WINDOW_S, "timing_tolerance_s": signals.TIMING_TOLERANCE_S,
        "qc": signals.QC, "regions": signals.REGIONS, "bands": signals.BANDS,
        "targets": trials.TARGETS, "development_ids": development_ids,
        "alphas": evaluation.ALPHAS, "outer_folds_max": evaluation.OUTER_FOLDS,
        "inner_folds_max": evaluation.INNER_FOLDS, "bootstrap_draws": evaluation.BOOTSTRAP_DRAWS,
        "bootstrap_seed": evaluation.SEED, "fallback_coverage_threshold": evaluation.FALLBACK_THRESHOLD,
        "fallback_modalities": evaluation.FALLBACK, "populations": populations,
        "markers_source": markers_source,
        "labels_path": str(labels_path.resolve()),
        "markers_path": str(markers_path.resolve()),
        "flagged_sensitivity": "évaluer les prédictions existantes sur les essais non signalés",
        "sample_smoke": smoke,
    }
    tables = {
        "manifest.csv": manifest, "coverage.csv": coverage, "features.csv": features,
        "folds.csv": folds, "predictions.csv": predictions, "metrics.csv": metrics,
    }
    if smoke:
        LOG.info("Contrôle de lecture sur les BDF de repos.")
    inspection = sample_inspection(root) if smoke else None
    LOG.info("Écriture des résultats dans %s", out_dir.resolve())
    write_outputs(out_dir, tables, config, inspection)
    LOG.info("Exécution terminée en %.1f s (%s).", monotonic() - started, status)
    return config
