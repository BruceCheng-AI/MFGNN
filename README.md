# TITS Traffic Congestion Prediction

This workspace contains the experimental code and release-ready documentation for a Shenzhen street-level traffic congestion prediction study. The prediction targets are `traffic_speed` and `TPI`, and the inputs combine historical traffic, weather, holiday, and event signals.

The current local folder still includes large datasets, cached tensors, trained checkpoints, and generated outputs. For public release, split it into:

- a public code repository
- a public dataset package or dataset repository

The release notes under `docs/open_source/` are prepared for that split.

## What To Publish In The Code Repository

- `notebooks/scripts/*.py`
- `notebooks/util.py`
- `requirements.txt`
- `docs/open_source/`
- `StreetSZ/config.json`

## What To Keep Out Of The Code Repository

- `data/raw/`
- `data/processed/`
- `StreetSZ/*.dyna`
- `StreetSZ/*.ext`
- `StreetSZ/*.fut`
- `StreetSZ/*.geo`
- `StreetSZ/*.his`
- `StreetSZ/*.rel`
- `StreetSZ/Street_Attr.csv`
- `StreetSZ/shp/`
- `cache/`
- `loss/`
- `output/`
- `model*/`
- `*.pt`
- `*.pth`
- `*.npy`
- TensorBoard event files

## Repository Layout

- `notebooks/scripts/`: core modeling, preprocessing, ablation, and meta-learning scripts
- `notebooks/util.py`: shared utility helpers
- `data/`: local raw and processed tables used during experiments
- `StreetSZ/`: LibCity-style dataset files and spatial metadata
- `docs/open_source/`: release notes for code publishing and dataset publishing
- `cache/`, `loss/`, `output/`, `model*`: local experiment artifacts and checkpoints

## Environment

Create a fresh environment and install the public dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducibility Notes

- The current project keeps the original experiment layout under `notebooks/`.
- Several scripts import modules with `from scripts...`, so keep the `notebooks/` layout unchanged or refactor imports before publishing.
- The public dataset package expected by the experiments is documented in `docs/open_source/DATASET_CARD.md`.
- The main public-facing code files should come from `notebooks/scripts/`, while heavy outputs and cached arrays should be released separately or omitted.

## Release Docs

- `docs/open_source/OPEN_SOURCE_SPLIT_PLAN.md`
- `docs/open_source/CODE_RELEASE_CHECKLIST.md`
- `docs/open_source/DATASET_RELEASE_CHECKLIST.md`
- `docs/open_source/DATASET_CARD.md`
- `docs/open_source/DATASET_DICTIONARY.md`

## Release Helper

Generate clean release folders from the current local workspace:

```bash
python prepare_open_source_release.py
```

This creates:

- `_open_source_release/code_repo`
- `_open_source_release/dataset_package`

## Before Publishing

1. Choose and add a code license.
2. Choose and add a dataset license.
3. Remove or relocate local artifacts covered by `.gitignore`.
4. Verify that every dataset source can be redistributed publicly.
5. Add the exact training and evaluation commands used in the paper or final report.
