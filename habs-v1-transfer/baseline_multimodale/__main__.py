"""Command line for the offline HABS baseline."""

import argparse
import logging
from pathlib import Path

from .pipeline import process


def main():
    parser = argparse.ArgumentParser(description="Baselines multimodales HABS V0 et V1")
    parser.add_argument("--data", default=Path("data"), type=Path, help="Racine du dataset HABS (par défaut ./data)")
    parser.add_argument("--csv-dir", type=Path, help="Dossier de labels_train.csv et markers.csv (par défaut --data)")
    parser.add_argument("--out", required=True, type=Path, help="Dossier des résultats")
    parser.add_argument("--smoke", action="store_true", help="Vérifier aussi le BDF de repos local")
    parser.add_argument("--development-id", action="append", default=[], help="ID du lot pilote à retirer de l’évaluation")
    parser.add_argument("--v0-dir", type=Path, help="Exécuter V1 sur les artefacts figés de ce dossier V0")
    parser.add_argument("--check-ecg", action="store_true", help="Pilote ECG V1 sur le premier pli d’activation")
    parser.add_argument("--cache-dir", type=Path, help="Cache partagé des fenêtres V1 en unités physiques")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    root = args.data.expanduser()
    if root.name == "PREPROCESSED" and root.is_dir():
        root = root.parent
    elif not (root / "PREPROCESSED").is_dir() and (root / "data" / "PREPROCESSED").is_dir():
        root = root / "data"
    if not (root / "PREPROCESSED").is_dir():
        parser.error(f"PREPROCESSED introuvable dans {root.resolve()}. Indiquer son dossier parent avec --data.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.v0_dir:
        if args.csv_dir or args.smoke or args.development_id:
            parser.error("--v0-dir utilise les essais et IDs déjà figés dans V0")
        from .v1 import process as run_v1
        try:
            result = run_v1(root, args.out, args.v0_dir, cache_dir=args.cache_dir,
                            device=args.device, pilot=args.check_ecg)
        except (OSError, ValueError) as exc:
            parser.exit(2, f"{exc}\n")
        print(f"V1 {result['status']} | {args.out.resolve()}")
        return 0
    if args.check_ecg or args.cache_dir or args.device != "auto":
        parser.error("Les options V1 demandent --v0-dir")
    csv_dir = args.csv_dir.expanduser() if args.csv_dir is not None else None
    try:
        config = process(root, args.out, csv_dir=csv_dir, smoke=args.smoke, development_ids=args.development_id)
    except PermissionError as exc:
        parser.exit(2, f"Accès refusé à {exc.filename or exc}. Vérifier les autorisations de l’application qui lance Python.\n")
    except FileNotFoundError as exc:
        parser.exit(2, f"{exc}\n")
    counts = config["input"]
    print(f"{config['status']} | {counts['video_bdfs']} BDF vidéo | "
          f"{counts['manifest_rows']} essais inventoriés | {counts['matched_trials']} fenêtres vérifiées")
    print(args.out.resolve())
    if not counts["labels_available"]:
        print(f"Table des notes manquante : {config['labels_path']}")
    if config["status"] == "insufficient_main_population":
        primary = config["populations"]["four_modalities"]["targets"]["rating_arousal"]
        if primary["n_ids"] < 3:
            print(f"{primary['n_ids']} ID(s) avec notes et quatre modalités exploitables. "
                  "La validation imbriquée en demande au moins trois.")
        else:
            print("Les notes et la couverture ne permettent pas de former les plis imbriqués.")
    return 0 if args.smoke or config["status"] == "evaluated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
