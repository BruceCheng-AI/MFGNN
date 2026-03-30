# MFGNN Package

This package is a minimal runnable extraction of the MFGNN pipeline used in the StreetSZ experiments.
It keeps only the core pieces needed to reproduce the training and evaluation workflow:

- StreetSZ data loading and sequence construction
- MFGNN model definition
- Reptile-style meta-training
- anomalous fine-tuning and scenario evaluation

## Package Structure

```text
project_root/
|-- StreetSZ/
|   |-- StreetSZ.geo
|   |-- StreetSZ.rel
|   |-- StreetSZ.dyna
|   |-- StreetSZ.ext
|   `-- StreetSZ.fut
`-- MFGNN_package/
    |-- run_mfgnn.py
    |-- run_full.bat
    |-- run_smoke_test.bat
    |-- requirements.txt
    `-- mfgnn/
        |-- __init__.py
        |-- data.py
        |-- model.py
        `-- train.py
```

## Requirements

- Python 3.10+
- PyTorch
- numpy
- pandas
- tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Directory

You can always pass the dataset directory explicitly:

```bash
python run_mfgnn.py --dataset-dir "E:\path\to\StreetSZ"
```

If `--dataset-dir` is not provided, the script now searches these locations automatically:

1. `../StreetSZ`
2. `./StreetSZ`
3. `../TITS2/StreetSZ`
4. `../TITS/StreetSZ`
5. `../dataset_package/StreetSZ`

If none of them contains a complete StreetSZ dataset, the script prints a clear error message with the searched paths.

## Quick Start

Run the default training + fine-tuning + evaluation pipeline:

```bash
python run_mfgnn.py
```

Or use the Windows batch wrapper:

```bat
run_full.bat
```

## Smoke Test

For a fast validation run on CPU:

```bat
run_smoke_test.bat
```

This uses a tiny configuration:

- `meta_epochs=1`
- `fine_tune_epochs=1`
- `num_tasks=1`
- `task_batch_size=1`
- `batch_size=2`
- `device=cpu`

You can still override arguments, for example:

```bat
run_smoke_test.bat --dataset-dir "E:\path\to\StreetSZ"
```

## Resume From a Meta Checkpoint

```bash
python run_mfgnn.py --meta-checkpoint ".\mfgnn_outputs\mfgnn_checkpoints\mfgnn_meta_epoch200.pt"
```

## Default Configuration

The default configuration follows the original notebook settings:

- `sequence_length=8`
- `forecast_horizon=4`
- `hidden_dim=64`
- `num_heads=4`
- `edge_hidden_dim=32`
- `num_layers=2`
- `dropout=0.1`
- `num_tasks=10`
- `support_ratio=0.8`
- `task_batch_size=4`
- `adapt_steps=2`
- `meta_lr=1e-3`
- `fine_tune_lr=5e-4`
- `meta_epochs=200`
- `fine_tune_epochs=15`

## Outputs

The pipeline writes results into `mfgnn_outputs/` by default:

- `mfgnn_checkpoints/mfgnn_meta_epoch*.pt`
- `mfgnn_checkpoints/mfgnn_final_finetuned.pt`
- `mfgnn_meta_loss_history.csv`
- `mfgnn_config.json`
- `mfgnn_metrics.json`

`mfgnn_metrics.json` contains:

- `full_test`
- `all_anomalous`
- `alert_weather`
- `holiday`
- `event`

Each scenario includes:

- `overall`
- `traffic_speed`
- `TPI`

## Validation Note

This package has been smoke-tested locally with:

- a minimal train/fine-tune/evaluate run
- resume-from-checkpoint execution

The exact training speed and final metrics will depend on your hardware, Python environment, and dataset location.
