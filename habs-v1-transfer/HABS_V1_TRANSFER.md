# Transfer the HABS multimodal V1 test

This archive contains the HABS entry point, the V0 and V1 implementation, the pinned Python requirements, the BDF loader and a fallback marker table. HABS provides the preprocessed recordings and training labels at runtime.

## Install on a branch in the new repository

Run these commands from the root of the new team's Git clone. Replace the ZIP path with the location on your machine.

```sh
git switch main
git pull --ff-only
git switch -c multimodal-v1
v1_stage=$(mktemp -d)
unzip /absolute/path/to/habs-v1-transfer.zip -d "$v1_stage"
```

Review the incoming entry point and dependency pins before copying.

```sh
git diff --no-index -- process.py "$v1_stage/process.py"
git diff --no-index -- requirements.txt "$v1_stage/requirements.txt"
```

Git returns 1 when these files differ. If the team's `process.py` is the starter and its requirements contain only template dependencies, you can install the files directly with the commands below. If the team has its own implementation or dependency pins, integrate the entry point and requirements with the team before replacing those files. If `baseline_multimodale/` already exists, review it before copying. The included BDF loader and marker table are at their original HABS template paths. Review any team edits to those files before copying them too.

```sh
cp -R "$v1_stage/baseline_multimodale" .
mkdir -p starting_notebooks
cp "$v1_stage/starting_notebooks/bdf_loader.py" "$v1_stage/starting_notebooks/markers.csv" starting_notebooks/
cp "$v1_stage/process.py" "$v1_stage/requirements.txt" "$v1_stage/HABS_V1_TRANSFER.md" .
```

Use a Python 3.12 environment to install the pinned dependencies and run the synthetic V1 tests.

```sh
python -m pip install -r requirements.txt
python -m unittest -v baseline_multimodale.test_v1
```

The V1 suite generates synthetic inputs. The V0 sample smoke test also needs the HABS sample files, if available in the new repo.

```sh
git add -- process.py requirements.txt HABS_V1_TRANSFER.md baseline_multimodale starting_notebooks/bdf_loader.py starting_notebooks/markers.csv
git diff --cached --check
git diff --cached --stat
git commit -m "Add HABS multimodal V1 evaluation"
git push -u origin multimodal-v1
```

Open a pull request into the new repository's `main` branch. After merging, run the new team's HABS workflow on `main` using its own team key. HABS calls the root `process.py` and mounts the dataset at `/data`. The entry point builds V0, then runs the short V1 configuration on CUDA with all seven models, both targets, all available folds and a maximum of three epochs per fit.

In the HABS results, check `v1/config.json` for the run status and open `v1/comparison.csv` for the scores. For a local run with the actual preprocessed dataset and its labels, use the following from the repository root.

```sh
HABS_BASELINE=v1 habs run --data /path/to/HABS --out ./out/habs-v1
```

Source revision: 94c5862
