# Open Source Split Plan

This workspace should be published as two separate public assets.

## 1. Public Code Repository

Purpose:
Release the modeling pipeline, experiment scripts, and documentation without bundling large local artifacts.

Keep:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `notebooks/scripts/*.py`
- `notebooks/util.py`
- `docs/open_source/`
- `StreetSZ/config.json`

Do not keep:

- `data/raw/`
- `data/processed/`
- `loss/`
- `cache/`
- `output/`
- `model*/`
- `*.pt`
- `*.pth`
- `*.npy`
- LibCity data tables under `StreetSZ/` except `config.json`

## 2. Public Dataset Package

Purpose:
Release the processed dataset needed for reproducible modeling, plus optional raw source tables and spatial files.

Recommended structure:

- `data/processed/`
- `StreetSZ/`
- `data/raw/` as an optional supplementary package
- `DATASET_CARD.md`
- `DATASET_DICTIONARY.md`
- dataset license file
- version and checksum notes

## Recommended Release Strategy

- Publish the code repository on GitHub or GitLab.
- Publish the dataset on Zenodo, Figshare, Kaggle, or an institutional repository.
- Link the dataset DOI or download page from the code repository README.
- If you want to publish pretrained weights, release them as a third optional asset instead of mixing them into the code repository.

## Minimum Public Deliverables

- code repository README
- dependency file
- code license
- dataset card
- dataset dictionary
- dataset license
- release checklist

## Final Sanity Check

- confirm redistribution rights for every raw source
- verify no confidential paths, lab-only outputs, or private annotations remain
- make sure the public repository can be cloned without downloading large local experiment artifacts
