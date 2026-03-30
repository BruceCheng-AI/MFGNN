from __future__ import annotations

import contextlib
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .data import SCENARIO_CHANNELS, TARGET_NAMES, anomalous_indices_from_dataset, scenario_indices_from_dataset


@dataclass
class MetaTrainConfig:
    meta_lr: float = 1e-3
    fast_lr: float = 1e-3
    epochs: int = 200
    task_batch_size: int = 4
    adapt_steps: int = 2
    max_norm: float = 1.0
    gcacs_threshold: float = 0.6
    gcacs_scaling: float = 0.6
    save_interval: int = 100


@dataclass
class FineTuneConfig:
    fast_lr: float = 5e-4
    epochs: int = 15
    patience: int = 5
    min_lr: float = 1e-6
    weight_decay: float = 1e-3
    mixed_precision: bool = True
    alpha: float = 0.8
    beta: float = 0.15
    gamma: float = 0.05


class Lion(Optimizer):
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign()
                param.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class TemporalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, beta: float = 0.15, gamma: float = 0.05) -> None:
        super().__init__()
        self.mae = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def temporal_smoothness_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return torch.mean((pred_diff - target_diff) ** 2)

    def trend_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        pred_sign = torch.sign(pred_diff)
        target_sign = torch.sign(target_diff)
        return torch.mean((pred_sign - target_sign) ** 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae_loss = self.mae(pred, target)
        smooth_loss = self.temporal_smoothness_loss(pred, target)
        trend_loss = self.trend_consistency_loss(pred, target)
        return self.alpha * mae_loss + self.beta * smooth_loss + self.gamma * trend_loss


def save_json(payload: Dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def subset_loader(dataset: Dataset, indices: Sequence[int], batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader | None:
    if not indices:
        return None
    subset = Subset(dataset, list(indices))
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_scenario_loaders(dataset: Dataset, batch_size: int, num_workers: int = 0) -> Dict[str, DataLoader | None]:
    scenario_loaders: Dict[str, DataLoader | None] = {
        "all_anomalous": subset_loader(
            dataset,
            anomalous_indices_from_dataset(dataset),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    }
    for scenario_name in SCENARIO_CHANNELS:
        scenario_loaders[scenario_name] = subset_loader(
            dataset,
            scenario_indices_from_dataset(dataset, scenario_name),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return scenario_loaders


def create_sequential_meta_tasks(
    dataset: Dataset,
    num_tasks: int = 10,
    support_ratio: float = 0.8,
    batch_size: int = 16,
    num_workers: int = 0,
) -> List[Tuple[DataLoader, DataLoader]]:
    total_samples = len(dataset)
    if total_samples < 2:
        raise ValueError("Not enough samples to create meta tasks.")

    task_size = max(1, total_samples // num_tasks)
    tasks: List[Tuple[DataLoader, DataLoader]] = []
    for task_id in range(num_tasks):
        start_idx = task_id * task_size
        end_idx = total_samples if task_id == num_tasks - 1 else min(total_samples, (task_id + 1) * task_size)
        task_indices = list(range(start_idx, end_idx))
        if len(task_indices) < 2:
            continue
        split_idx = int(len(task_indices) * support_ratio)
        split_idx = max(1, min(split_idx, len(task_indices) - 1))
        support_indices = task_indices[:split_idx]
        query_indices = task_indices[split_idx:]

        support_loader = subset_loader(dataset, support_indices, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        query_loader = subset_loader(dataset, query_indices, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if support_loader is not None and query_loader is not None:
            tasks.append((support_loader, query_loader))

    if not tasks:
        raise ValueError("No valid meta tasks were created.")
    return tasks


def _build_named_adamw(model: nn.Module, lr: float) -> torch.optim.AdamW:
    param_groups = [{"params": [param], "name": name, "lr": lr} for name, param in model.named_parameters()]
    return torch.optim.AdamW(param_groups, lr=lr)


def compute_gradient_cosine_similarity(model: nn.Module) -> Dict[str, Dict[str, float]]:
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad.reshape(-1)

    layer_names = list(gradients.keys())
    cosine_similarities = {name: {} for name in layer_names}
    for index_i, name_i in enumerate(layer_names):
        grad_i = gradients[name_i]
        for index_j, name_j in enumerate(layer_names):
            if index_i >= index_j:
                continue
            grad_j = gradients[name_j]
            if grad_i.shape != grad_j.shape:
                continue
            if grad_i.norm() == 0 or grad_j.norm() == 0:
                cosine_similarity = 0.0
            else:
                cosine_similarity = torch.nn.functional.cosine_similarity(grad_i, grad_j, dim=0).item()
            cosine_similarities[name_i][name_j] = cosine_similarity
            cosine_similarities[name_j][name_i] = cosine_similarity
    return cosine_similarities


def adjust_learning_rates_gcacs(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    threshold: float,
    scaling_factor: float,
    base_lr: float,
) -> None:
    cosine_similarities = compute_gradient_cosine_similarity(model)
    scaling_factors = {name: 1.0 for name, _ in model.named_parameters()}
    for name, similarities in cosine_similarities.items():
        for _, similarity in similarities.items():
            if similarity < threshold:
                scaling_factors[name] *= scaling_factor

    for param_group in optimizer.param_groups:
        name = param_group.get("name")
        if name in scaling_factors:
            param_group["lr"] = base_lr * scaling_factors[name]


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _forward_batch(model: nn.Module, batch: Dict[str, torch.Tensor], adjacency_matrix: torch.Tensor) -> torch.Tensor:
    return model(
        dyna_traffic=batch["dyna_traffic"],
        ext_weather=batch["ext_weather"],
        time_seq=batch["time_seq"],
        ind_seq=batch["ind_seq"],
        ind_hor=batch["ind_hor"],
        adjacency_matrix=adjacency_matrix,
        node_array=batch["node"],
        time_hor=batch["time_hor"],
    )


def meta_train_reptile(
    model: nn.Module,
    tasks: Sequence[Tuple[DataLoader, DataLoader]],
    device: torch.device,
    adjacency_matrix: torch.Tensor,
    output_dir: Path,
    config: MetaTrainConfig,
) -> Tuple[nn.Module, List[float]]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "mfgnn_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.L1Loss()
    meta_optimizer = _build_named_adamw(model, lr=config.meta_lr)
    meta_loss_history: List[float] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        selected_tasks = random.sample(list(tasks), min(config.task_batch_size, len(tasks)))
        delta_accumulators = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
        total_meta_loss = 0.0

        for support_loader, query_loader in selected_tasks:
            task_model = copy.deepcopy(model).to(device)
            task_optimizer = torch.optim.AdamW(task_model.parameters(), lr=config.fast_lr)
            task_loss = 0.0

            for _ in range(config.adapt_steps):
                combined_batches: Iterable[Dict[str, torch.Tensor]] = list(support_loader) + list(query_loader)
                for batch in combined_batches:
                    batch = _move_batch_to_device(batch, device)
                    task_optimizer.zero_grad(set_to_none=True)
                    outputs = _forward_batch(task_model, batch, adjacency_matrix)
                    loss = nn.L1Loss()(outputs, batch["target"])
                    loss.backward()
                    task_optimizer.step()
                    task_loss = float(loss.item())

            with torch.no_grad():
                task_state = task_model.state_dict()
                for name, param in model.named_parameters():
                    delta_accumulators[name].add_(param.data - task_state[name].data)
            total_meta_loss += task_loss

        meta_optimizer.zero_grad(set_to_none=True)
        for name, param in model.named_parameters():
            param.grad = delta_accumulators[name] / max(1, len(selected_tasks))
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        adjust_learning_rates_gcacs(
            model=model,
            optimizer=meta_optimizer,
            threshold=config.gcacs_threshold,
            scaling_factor=config.gcacs_scaling,
            base_lr=config.meta_lr,
        )
        meta_optimizer.step()

        avg_meta_loss = total_meta_loss / max(1, len(selected_tasks))
        meta_loss_history.append(float(avg_meta_loss))

        if epoch % config.save_interval == 0 or epoch == config.epochs:
            checkpoint_path = checkpoint_dir / f"mfgnn_meta_epoch{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)

    history_csv = output_dir / "mfgnn_meta_loss_history.csv"
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "meta_loss"])
        for epoch, loss in enumerate(meta_loss_history, start=1):
            writer.writerow([epoch, loss])

    save_json(
        {
            "meta_train_config": asdict(config),
            "meta_loss_history": meta_loss_history,
        },
        output_dir / "mfgnn_meta_history.json",
    )
    return model, meta_loss_history


def fine_tune_model(
    model: nn.Module,
    fine_tune_loader: DataLoader,
    device: torch.device,
    adjacency_matrix: torch.Tensor,
    config: FineTuneConfig,
) -> Tuple[nn.Module, List[float]]:
    if fine_tune_loader is None or len(fine_tune_loader) == 0:
        raise ValueError("fine_tune_loader is empty.")

    criterion = TemporalLoss(alpha=config.alpha, beta=config.beta, gamma=config.gamma)
    optimizer = Lion(model.parameters(), lr=config.fast_lr * 0.1, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(fine_tune_loader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.fast_lr,
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy="cos",
    )

    amp_enabled = config.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_loss = float("inf")
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())
    train_losses: List[float] = []

    for _ in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        progress = tqdm(fine_tune_loader, desc="Fine-tune", leave=False)
        for batch in progress:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
            with autocast_context:
                outputs = _forward_batch(model, batch, adjacency_matrix)
                loss = criterion(outputs, batch["target"])

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            scheduler.step()
            batch_loss = float(loss.item())
            epoch_loss += batch_loss
            batch_count += 1
            progress.set_postfix(loss=batch_loss, lr=optimizer.param_groups[0]["lr"])

        avg_epoch_loss = epoch_loss / max(1, batch_count)
        train_losses.append(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    model.load_state_dict(best_state)
    return model, train_losses


def _denormalize_outputs(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    stats: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_mean = torch.tensor(stats["target_mean"], dtype=torch.float32, device=device).unsqueeze(0)
    target_std = torch.tensor(stats["target_std"], dtype=torch.float32, device=device).unsqueeze(0)
    outputs_original = outputs * target_std + target_mean
    targets_original = targets * target_std + target_mean
    return outputs_original, targets_original


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    outputs_np = outputs.reshape(-1).cpu().numpy()
    targets_np = targets.reshape(-1).cpu().numpy()
    diff = targets_np - outputs_np
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(diff) / (targets_np + 0.1)) * 100.0)
    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    adjacency_matrix: torch.Tensor,
    stats: Dict[str, np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if data_loader is None or len(data_loader) == 0:
        raise ValueError("data_loader is empty.")

    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluate", leave=False):
            batch = _move_batch_to_device(batch, device)
            outputs = _forward_batch(model, batch, adjacency_matrix)
            outputs_original, targets_original = _denormalize_outputs(outputs, batch["target"], stats, device)
            all_outputs.append(outputs_original.cpu())
            all_targets.append(targets_original.cpu())

    return torch.cat(all_outputs, dim=0), torch.cat(all_targets, dim=0)


def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    adjacency_matrix: torch.Tensor,
    stats: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    outputs, targets = collect_predictions(model, data_loader, device, adjacency_matrix, stats)
    metrics = {"overall": compute_metrics(outputs, targets)}
    for target_index, target_name in enumerate(TARGET_NAMES):
        metrics[target_name] = compute_metrics(outputs[:, :, :, target_index], targets[:, :, :, target_index])
    return metrics


def evaluate_scenario_loaders(
    model: nn.Module,
    scenario_loaders: Dict[str, DataLoader | None],
    device: torch.device,
    adjacency_matrix: torch.Tensor,
    stats: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics = {}
    for scenario_name, loader in scenario_loaders.items():
        if loader is None:
            continue
        metrics[scenario_name] = evaluate_loader(
            model=model,
            data_loader=loader,
            device=device,
            adjacency_matrix=adjacency_matrix,
            stats=stats,
        )
    return metrics

