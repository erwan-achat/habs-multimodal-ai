"""Train V1 against frozen V0 folds. Each completed inner fit and external model is resumable."""

import fcntl
import hashlib
import itertools
import json
import logging
import os
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import torch

from . import evaluation
from .pipeline import write_outputs
from .signals import FEATURE_COLUMNS
from .trials import KEY, TARGETS
from .v1_data import EEG, RATES, REFERENCES, atomic_file, digest_file, frozen_study, load_windows, read_table
from .v1_model import Banks, MODELS, Regressor

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Training:
    epochs: int = 30
    batch_size: int = 32
    rates: tuple = (0.0003, 0.001)
    decays: tuple = (0.0001, 0.01)
    seed: int = 2026


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


@torch.inference_mode()
def predict(model, banks, indices, batch_size):
    model.eval()
    return torch.cat([banks.forward(model, batch) for batch in indices.split(batch_size)]).cpu().numpy()


def fit(name, banks, labels, train, validation, rate, decay, epochs, settings):
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    model = Regressor(name)
    banks.configure(model, train)
    model.to(banks.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
    validation_predictions = []
    if validation is not None:
        validation = torch.tensor(validation, device=banks.device)
    for _ in range(epochs):
        model.train()
        shuffled = torch.as_tensor(rng.permutation(train), device=banks.device)
        for batch in shuffled.split(settings.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(banks.forward(model, batch), labels[batch])
            loss.backward()
            optimizer.step()
        if validation is not None:
            validation_predictions.append(predict(model, banks, validation, settings.batch_size))
    return model, np.asarray(validation_predictions)


def save_json(path, value):
    with atomic_file(path) as temp:
        temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def select_training(name, banks, rows, target, labels, inner, folder, settings):
    history = []
    for rate, decay in itertools.product(settings.rates, settings.decays):
        id_errors = []
        for number, (train, val) in enumerate(inner, 1):
            path = folder / f"inner-{number}-lr-{rate}-wd-{decay}.npy"
            if path.exists():
                errors = np.load(path, allow_pickle=False)
            else:
                _, preds = fit(name, banks, labels, train, val, rate, decay, settings.epochs, settings)
                absolute = np.abs(preds.T - rows.loc[val, target].to_numpy()[:, None])
                errors = pd.DataFrame(absolute, index=rows.loc[val, "participant_id"]).groupby(level=0).mean().to_numpy()
                if not np.isfinite(errors).all():
                    raise ValueError(f"{target}/{name}: non-finite validation error")
                with atomic_file(path) as temp, temp.open("wb") as handle:
                    np.save(handle, errors)
            id_errors.append(errors)
        scores = np.concatenate(id_errors).mean(axis=0)
        history.extend((float(score), epoch, rate, decay) for epoch, score in enumerate(scores, 1))
    score, epoch, rate, decay = min(history)
    return dict(inner_mae=score, epoch=epoch, learning_rate=rate, weight_decay=decay)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Regressor(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def run_fold(name, banks, rows, target, labels, split, folder, settings):
    train, test, inner = split
    path = folder / "model.pt"
    if not path.exists():
        choice = select_training(name, banks, rows, target, labels, inner, folder, settings)
        model, _ = fit(name, banks, labels, train, None, choice["learning_rate"], choice["weight_decay"],
                       choice["epoch"], settings)
        with atomic_file(path) as temp:
            torch.save({"model": name, "target": target, "selection": choice,
                        "train_ids": sorted(rows.loc[train, "participant_id"].unique().tolist()),
                        "state_dict": model.cpu().state_dict()}, temp)
    model, saved = load_model(path, banks.device)
    indices = torch.tensor(test, device=banks.device)
    pred = predict(model, banks, indices, settings.batch_size)
    if not np.isfinite(pred).all():
        raise ValueError(f"{target}/{name}: non-finite external prediction")
    return pred, saved["selection"]


def comparison_table(metrics, coverage, manifest, fusion, protocol):
    rows = []
    for target, result in metrics[metrics.subset.eq("all")].groupby("target"):
        models = result[result.kind.eq("model")].set_index("name")
        gains = result[result.kind.eq("paired_gain")].set_index("name")
        stages = coverage[coverage.cohort.eq("all") & coverage.target.eq(target)].set_index("stage")
        matched = manifest[manifest.trial_status.eq("ok") & manifest[target].notna()]
        cycles = matched.resp_issues.eq("resp:insufficient_cycles")
        only_resp = cycles & matched[["eeg_status", "ecg_status", "gsr_status"]].eq("ok").all(axis=1)
        ridge = gains.loc[f"four_modalities_vs_{fusion}"]
        median = gains.loc[f"median_vs_{fusion}"]
        rows.append(dict(protocol=protocol, target=target, model=fusion,
                         matched_labelled=int(stages.loc["matched_trial", "n_trials"]),
                         eligible=int(stages.loc["four_modalities", "n_trials"]),
                         eligible_fraction=stages.loc["four_modalities", "n_trials"] / len(matched),
                         resp_insufficient_cycles=int(cycles.sum()), resp_only_exclusion=int(only_resp.sum()),
                         ecg_gsr_trials=int(stages.loc["ecg_gsr", "n_trials"]),
                         ecg_gsr_ids=int(stages.loc["ecg_gsr", "n_ids"]),
                         n_test=int(models.loc[fusion, "n_trials"]), n_ids=int(models.loc[fusion, "n_ids"]),
                         mae_macro=models.loc[fusion, "mae_macro"], ridge_mae=models.loc["four_modalities", "mae_macro"],
                         gain_vs_ridge=ridge.gain, ci_low=ridge.ci_low, ci_high=ridge.ci_high,
                         direction_vs_ridge=ridge.direction, median_mae=models.loc["median", "mae_macro"],
                         gain_vs_median=median.gain, direction_vs_median=median.direction,
                         clip_ridge_mae=models.loc["clip_ridge", "mae_macro"],
                         clip_seen_fraction=models.loc[fusion, "clip_seen_fraction"]))
    return pd.DataFrame(rows)


def process(data_dir, out_dir, v0_dir, *, cache_dir=None, device="auto", pilot=False, settings=None):
    out_dir, v0_dir = Path(out_dir).resolve(), Path(v0_dir).resolve()
    if out_dir == v0_dir or (out_dir / "manifest.csv").exists():
        raise ValueError("Choisir un dossier V1 distinct des artefacts V0")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ".v1.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Une exécution utilise déjà {out_dir}") from exc
        return _study(Path(data_dir), out_dir, v0_dir, cache_dir, choose_device(device), pilot, settings)


def _study(root, out_dir, v0_dir, cache_dir, device, pilot, settings):
    v0 = json.loads((v0_dir / "config.json").read_text())
    if v0["status"] != "evaluated":
        raise ValueError("V0 doit avoir évalué la population four_modalities")
    rows, partitions, references = frozen_study(v0_dir)
    models = ("v1_ecg",) if pilot else tuple(MODELS)
    if pilot:
        key = min(k for k in partitions if k[0] == TARGETS[0])
        partitions = {key: partitions[key]}
    if settings is None:
        settings = Training(epochs=1, rates=(0.001,), decays=(0.0001,)) if pilot else Training()
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    LOG.info("V1 sur %s, %d essais retenus. Lecture du cache par session.", device, len(rows))
    windows, lengths, session_hash = load_windows(root, Path(cache_dir or out_dir / "signal_cache"), rows)
    banks = Banks(windows, lengths, rows, device)
    del windows
    filenames = ("manifest.csv", "features.csv", "folds.csv", "predictions.csv", "coverage.csv", "config.json")
    modules = ("v1.py", "v1_model.py", "v1_data.py", "evaluation.py", "signals.py", "trials.py")
    config = {
        "protocol": "ecg_pilot" if pilot else "v1", "settings": asdict(settings), "device": device,
        "models": list(models), "channels": list(EEG), "rates_hz": RATES, "window_s": v0["window_s"],
        "descriptor_columns": FEATURE_COLUMNS, "units": {"eeg": "uV", "ecg": "mV", "gsr": "uS", "resp": "AU"},
        "versions": {p: version(p) for p in ("torch", "numpy", "pandas", "scipy", "pyedflib")},
        "v0_data_signature": v0["data_signature"], "v0_code_sha256": v0["code_sha256"],
        "v0_artifacts": {f: digest_file(v0_dir / f) for f in filenames}, "window_signature": session_hash,
        "code_sha256": {f: digest_file(Path(__file__).with_name(f)) for f in modules},
        "bootstrap_seed": evaluation.SEED, "bootstrap_draws": evaluation.BOOTSTRAP_DRAWS,
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    folder = out_dir / "checkpoints" / signature
    config.update(signature=signature, status="running")
    old = out_dir / "config.json"
    if old.is_file() and json.loads(old.read_text()).get("signature") != signature:
        raise ValueError("Configuration V1 différente dans ce dossier de sortie. Choisir un autre --out.")
    save_json(old, config)
    predictions, choices = [], []
    for (target, fold), split in partitions.items():
        labels = torch.tensor(rows[target].to_numpy(), device=device, dtype=torch.float32)
        test = split[1]
        refs = references[references.target.eq(target) & references.outer_fold.eq(fold)]
        predictions.append(refs)
        reference = refs[refs.model.eq("four_modalities")].set_index(KEY)
        template = reference.loc[pd.MultiIndex.from_frame(rows.loc[test, KEY])].reset_index()
        for name in models:
            started = monotonic()
            location = folder / target / f"fold-{fold}" / name
            LOG.info("%s pli %d / %s", target, fold, name)
            values, choice = run_fold(name, banks, rows, target, labels, split, location, settings)
            predictions.append(template.assign(model=name, prediction=values, alpha=np.nan, selected_sensor=None))
            choices.append(dict(target=target, outer_fold=fold, model=name, checkpoint=str(
                (location / "model.pt").relative_to(out_dir)), **choice))
            LOG.info("Terminé en %.1f s", monotonic() - started)

    predictions = pd.concat(predictions, ignore_index=True)
    fusion = "v1_ecg" if pilot else "v1_four_modalities"
    contrasts = [(name, fusion) for name in REFERENCES]
    if not pilot:
        contrasts.extend([("descriptor_mlp", fusion), ("v1_peripheral", fusion)])
    metrics = evaluation.summarize(predictions, contrasts)
    summary = metrics.query("subset == 'all' and kind == 'model'")[
        ["target", "name", "n_ids", "n_trials", "mae_macro", "mae_trials", "clip_seen_fraction"]].copy()
    config.update(status="pilot" if pilot else "evaluated", contrasts=contrasts)
    coverage = read_table(v0_dir / "coverage.csv")
    manifest = read_table(v0_dir / "manifest.csv")
    manifest = manifest.loc[~manifest.participant_id.isin(v0["development_ids"])]
    tables = {"predictions.csv": predictions, "metrics.csv": metrics, "scores.csv": summary,
              "selection.csv": pd.DataFrame(choices), "coverage.csv": coverage,
              "comparison.csv": comparison_table(metrics, coverage, manifest, fusion, config["protocol"])}
    write_outputs(out_dir, tables, config, None)
    return config
