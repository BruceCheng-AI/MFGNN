# Multimodal Graph Meta-Learning for Few-Shot Traffic Forecasting under Anomalous Events

## Abstract

Accurate traffic forecasting under anomalous urban events remains challenging because extreme weather, public holidays, and large-scale gatherings can shift traffic patterns away from routine regimes, while event-specific samples are often limited. This study formulates anomalous-event traffic forecasting as a few-shot multimodal spatiotemporal prediction problem under distribution shift and proposes a graph meta-learning framework for rapid adaptation across heterogeneous scenarios. The framework integrates multimodal feature encoding, adaptive graph learning, and meta-learned parameter initialization to capture traffic dynamics, exogenous context, and event-driven changes in regional dependency. Experiments on StreetSZ, a real-world multimodal dataset covering 74 sub-districts in Shenzhen, show that the proposed method consistently outperforms representative statistical, sequential, and graph-based baselines in both overall and event-wise evaluations. Few-shot adaptation and ablation studies further support the contribution of the proposed design. The results indicate that graph-based rapid adaptation is a practical approach to robust traffic forecasting under anomalous urban conditions.

## Repository Overview

This repository contains the public code release for the MFGNN framework. The project focuses on multimodal traffic forecasting with street-level traffic targets and exogenous signals such as weather, holidays, and major events.

## Repository Structure

- `notebooks/scripts/`: model definitions, preprocessing, ablation, and meta-learning scripts
- `notebooks/util.py`: utility helpers
- `StreetSZ/config.json`: dataset schema reference
- `docs/open_source/`: release notes and dataset documentation

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Notes

- The current public code keeps the original experiment layout under `notebooks/`.
- Several scripts import modules with `from scripts ...`, so keep the existing folder layout when running the code.
- The dataset description is documented in `docs/open_source/DATASET_CARD.md` and `docs/open_source/DATASET_DICTIONARY.md`.

## Release Helper

```bash
python prepare_open_source_release.py
```
