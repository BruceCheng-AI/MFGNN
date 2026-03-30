from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    import numpy as np
    import torch

    from mfgnn import (
        FineTuneConfig,
        MFGNN,
        MetaTrainConfig,
        anomalous_indices_from_dataset,
        build_data_bundle,
        build_scenario_loaders,
        create_sequential_meta_tasks,
        evaluate_loader,
        evaluate_scenario_loaders,
        fine_tune_model,
        meta_train_reptile,
        save_json,
        subset_loader,
    )
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


STREETSZ_FILES = (
    "StreetSZ.geo",
    "StreetSZ.rel",
    "StreetSZ.dyna",
    "StreetSZ.ext",
    "StreetSZ.fut",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal meta-learning pipeline for StreetSZ + MFGNN.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--meta-checkpoint", type=Path, default=None)

    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--forecast-horizon", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.3)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--edge-hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--support-ratio", type=float, default=0.8)
    parser.add_argument("--meta-lr", type=float, default=1e-3)
    parser.add_argument("--fast-lr", type=float, default=1e-3)
    parser.add_argument("--meta-epochs", type=int, default=200)
    parser.add_argument("--task-batch-size", type=int, default=4)
    parser.add_argument("--adapt-steps", type=int, default=2)
    parser.add_argument("--gcacs-threshold", type=float, default=0.6)
    parser.add_argument("--gcacs-scaling", type=float, default=0.6)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=100)

    parser.add_argument("--fine-tune-lr", type=float, default=5e-4)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--disable-amp", action="store_true")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_streetsz_dataset_dir(path: Path) -> bool:
    path = Path(path)
    return all((path / file_name).exists() for file_name in STREETSZ_FILES)


def resolve_dataset_dir(package_root: Path, dataset_arg: Path | None) -> Path:
    if dataset_arg is not None:
        dataset_dir = Path(dataset_arg).expanduser().resolve()
        if is_streetsz_dataset_dir(dataset_dir):
            return dataset_dir
        missing = [str(dataset_dir / file_name) for file_name in STREETSZ_FILES if not (dataset_dir / file_name).exists()]
        raise FileNotFoundError(
            "The provided --dataset-dir does not contain a complete StreetSZ dataset.\n"
            + "Missing files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    workspace_root = package_root.parent
    candidate_dirs = [
        workspace_root / "StreetSZ",
        package_root / "StreetSZ",
        workspace_root / "TITS2" / "StreetSZ",
        workspace_root / "TITS" / "StreetSZ",
        workspace_root / "dataset_package" / "StreetSZ",
    ]
    for candidate_dir in candidate_dirs:
        if is_streetsz_dataset_dir(candidate_dir):
            return candidate_dir.resolve()

    searched = "\n".join(f"  - {candidate_dir}" for candidate_dir in candidate_dirs)
    raise FileNotFoundError(
        "Could not automatically locate the StreetSZ dataset.\n"
        "Searched these directories:\n"
        f"{searched}\n"
        'Please rerun with --dataset-dir "path\\to\\StreetSZ".'
    )


def main() -> None:
    args = parse_args()
    if IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing Python dependency while starting MFGNN_package: "
            f"{IMPORT_ERROR.name}\n"
            "Install dependencies first with: pip install -r requirements.txt"
        )

    package_root = Path(__file__).resolve().parent

    dataset_dir = resolve_dataset_dir(package_root, args.dataset_dir)
    output_dir = args.output_dir or (package_root / "mfgnn_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = choose_device(args.device)

    print(f"Using dataset directory: {dataset_dir}")
    print(f"Using output directory: {output_dir}")
    print(f"Using device: {device}")

    bundle = build_data_bundle(
        dataset_dir=dataset_dir,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    model = MFGNN(
        node_feature_dim=bundle.dims["node_feature_dim"],
        var_dim_dyna=bundle.dims["var_dim_dyna"],
        var_dim_weather=bundle.dims["var_dim_weather"],
        event_dim=bundle.dims["event_dim"],
        time_dim=bundle.dims["time_dim"],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        target_dim=bundle.dims["target_dim"],
        edge_hidden_dim=args.edge_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    adjacency_matrix = torch.tensor(bundle.adjacency_matrix, dtype=torch.float32, device=device)

    meta_config = MetaTrainConfig(
        meta_lr=args.meta_lr,
        fast_lr=args.fast_lr,
        epochs=args.meta_epochs,
        task_batch_size=args.task_batch_size,
        adapt_steps=args.adapt_steps,
        max_norm=args.max_norm,
        gcacs_threshold=args.gcacs_threshold,
        gcacs_scaling=args.gcacs_scaling,
        save_interval=args.save_interval,
    )

    if args.meta_checkpoint is not None:
        model.load_state_dict(torch.load(args.meta_checkpoint, map_location=device))
        meta_loss_history = []
    else:
        tasks = create_sequential_meta_tasks(
            dataset=bundle.train_dataset,
            num_tasks=args.num_tasks,
            support_ratio=args.support_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model, meta_loss_history = meta_train_reptile(
            model=model,
            tasks=tasks,
            device=device,
            adjacency_matrix=adjacency_matrix,
            output_dir=output_dir,
            config=meta_config,
        )

    anomalous_train_loader = subset_loader(
        bundle.train_dataset,
        anomalous_indices_from_dataset(bundle.train_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    fine_tune_config = FineTuneConfig(
        fast_lr=args.fine_tune_lr,
        epochs=args.fine_tune_epochs,
        patience=args.fine_tune_patience,
        weight_decay=args.weight_decay,
        mixed_precision=not args.disable_amp,
    )
    model, fine_tune_history = fine_tune_model(
        model=model,
        fine_tune_loader=anomalous_train_loader,
        device=device,
        adjacency_matrix=adjacency_matrix,
        config=fine_tune_config,
    )

    checkpoint_dir = output_dir / "mfgnn_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = checkpoint_dir / "mfgnn_final_finetuned.pt"
    torch.save(model.state_dict(), final_checkpoint)

    metrics = {
        "full_test": evaluate_loader(
            model=model,
            data_loader=bundle.test_loader,
            device=device,
            adjacency_matrix=adjacency_matrix,
            stats=bundle.stats,
        )
    }
    scenario_loaders = build_scenario_loaders(
        dataset=bundle.test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metrics.update(
        evaluate_scenario_loaders(
            model=model,
            scenario_loaders=scenario_loaders,
            device=device,
            adjacency_matrix=adjacency_matrix,
            stats=bundle.stats,
        )
    )

    save_json(
        {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "device": str(device),
            "seed": args.seed,
            "data_config": {
                "sequence_length": args.sequence_length,
                "forecast_horizon": args.forecast_horizon,
                "batch_size": args.batch_size,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
            },
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "edge_hidden_dim": args.edge_hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                **bundle.dims,
            },
            "meta_train_config": asdict(meta_config),
            "fine_tune_config": asdict(fine_tune_config),
            "meta_loss_history": meta_loss_history,
            "fine_tune_loss_history": fine_tune_history,
            "final_checkpoint": str(final_checkpoint),
        },
        output_dir / "mfgnn_config.json",
    )
    save_json(metrics, output_dir / "mfgnn_metrics.json")

    print("Pipeline artifacts saved to:")
    print(f"  {output_dir}")
    print("Scenario metrics:")
    for scenario_name, scenario_metrics in metrics.items():
        print(f"  {scenario_name}: {scenario_metrics['overall']}")


if __name__ == "__main__":
    main()


