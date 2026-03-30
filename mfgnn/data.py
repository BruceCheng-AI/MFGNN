from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

EPS = 1e-8
TARGET_NAMES = ("traffic_speed", "TPI")
SCENARIO_CHANNELS = {
    "alert_weather": 0,
    "holiday": 1,
    "event": 2,
}


@dataclass
class DataBundle:
    dataset_dir: Path
    train_dataset: "TrafficDataset"
    val_dataset: "TrafficDataset"
    test_dataset: "TrafficDataset"
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    adjacency_matrix: np.ndarray
    stats: Dict[str, np.ndarray]
    node_stats: Dict[str, np.ndarray]
    dims: Dict[str, int]


def _safe_std(values: np.ndarray) -> np.ndarray:
    return np.where(np.abs(values) < EPS, 1.0, values)


def read_streetsz_tables(dataset_dir: Path) -> Tuple[pd.DataFrame, ...]:
    dataset_dir = Path(dataset_dir)
    required = [
        dataset_dir / "StreetSZ.geo",
        dataset_dir / "StreetSZ.rel",
        dataset_dir / "StreetSZ.dyna",
        dataset_dir / "StreetSZ.ext",
        dataset_dir / "StreetSZ.fut",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing StreetSZ files: {missing}")

    geo = pd.read_csv(required[0])
    rel = pd.read_csv(required[1])
    dyna = pd.read_csv(required[2])
    ext = pd.read_csv(required[3])
    fut = pd.read_csv(required[4])
    return geo, rel, dyna, ext, fut


def read_and_process(
    geo: pd.DataFrame,
    rel: pd.DataFrame,
    dyna: pd.DataFrame,
    ext: pd.DataFrame,
    fut: pd.DataFrame,
) -> Tuple[np.ndarray, ...]:
    dyna = dyna.copy()
    ext = ext.copy()
    fut = fut.copy()

    dyna["time"] = pd.to_datetime(dyna["time"], utc=True)
    dyna["time_unix"] = dyna["time"].astype("int64") // 10**9
    dyna = dyna.sort_values(by=["time", "entity_id"])
    dyna_features = ["traffic_speed", "TPI", "time_unix"]

    time_steps = dyna["time"].nunique()
    nodes = dyna["entity_id"].nunique()
    dyna_array = np.zeros((time_steps, nodes, len(dyna_features)), dtype=np.float32)
    for index, (_, group) in enumerate(dyna.groupby("time", sort=True)):
        dyna_array[index, :, :] = group[dyna_features].to_numpy(dtype=np.float32)
    dyna_traffic = dyna_array[:, :, :2]
    time_unix = dyna_array[:, :, -1]

    ext["time"] = pd.to_datetime(ext["time"], utc=True)
    ext = ext.sort_values(by=["time", "geo_id"])
    ext_features = ["R1h", "W1h", "T1h", "V1h", "alert_level", "holiday_status", "event_rating"]
    time_steps = ext["time"].nunique()
    nodes = ext["geo_id"].nunique()
    ext_array = np.zeros((time_steps, nodes, len(ext_features)), dtype=np.float32)
    for index, (_, group) in enumerate(ext.groupby("time", sort=True)):
        ext_array[index, :, :] = group[ext_features].to_numpy(dtype=np.float32)
    ext_weather = ext_array[:, :, :4]
    ind_seq = ext_array[:, :, -3:]

    fut["time"] = pd.to_datetime(fut["time"], utc=True)
    fut["time_unix"] = fut["time"].astype("int64") // 10**9
    fut = fut.sort_values(by=["time", "geo_id"])
    fut_features = ["weather_forecast", "holiday_status", "event_rating"]
    time_steps = fut["time"].nunique()
    nodes = fut["geo_id"].nunique()
    ind_hor = np.zeros((time_steps, nodes, len(fut_features)), dtype=np.float32)
    for index, (_, group) in enumerate(fut.groupby("time", sort=True)):
        ind_hor[index, :, :] = group[fut_features].to_numpy(dtype=np.float32)

    num_nodes = int(max(rel["origin_id"].max(), rel["destination_id"].max()))
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in rel.iterrows():
        origin = int(row["origin_id"]) - 1
        destination = int(row["destination_id"]) - 1
        weight = float(row["link_weight"])
        adjacency_matrix[origin, destination] = weight
        adjacency_matrix[destination, origin] = weight

    node_columns = [
        "DISTRICT_ID",
        "Area",
        "RoadDensity",
        "BuildingArea",
        "CarStation",
        "CarPark",
        "Subway",
        "POI",
    ]
    node_array = geo[node_columns].to_numpy(dtype=np.float32)

    return dyna_traffic, time_unix, ext_weather, ind_seq, ind_hor, adjacency_matrix, node_array


def create_sequences(
    dyna_traffic: np.ndarray,
    time_unix: np.ndarray,
    ext_weather: np.ndarray,
    ind_seq: np.ndarray,
    ind_hor: np.ndarray,
    sequence_length: int,
    forecast_horizon: int,
) -> Dict[str, np.ndarray]:
    total_steps = dyna_traffic.shape[0] - sequence_length - forecast_horizon + 1
    if total_steps <= 0:
        raise ValueError("sequence_length + forecast_horizon exceeds available data length.")

    dyna_traffic_seq: List[np.ndarray] = []
    time_seq: List[np.ndarray] = []
    ext_weather_seq: List[np.ndarray] = []
    ind_seq_seq: List[np.ndarray] = []
    ind_hor_seq: List[np.ndarray] = []
    time_hor: List[np.ndarray] = []
    target_seq: List[np.ndarray] = []

    for start in range(total_steps):
        end = start + sequence_length
        horizon_end = end + forecast_horizon
        dyna_traffic_seq.append(dyna_traffic[start:end, :, :])
        time_seq.append(time_unix[start:end, :])
        ext_weather_seq.append(ext_weather[start:end, :, :])
        ind_seq_seq.append(ind_seq[start:end, :, :])
        ind_hor_seq.append(ind_hor[end:horizon_end, :, :])
        time_hor.append(time_unix[end:horizon_end, :])
        target_seq.append(dyna_traffic[end:horizon_end, :, :])

    return {
        "dyna_traffic": np.asarray(dyna_traffic_seq, dtype=np.float32),
        "time_seq": np.asarray(time_seq, dtype=np.float32),
        "ext_weather_seq": np.asarray(ext_weather_seq, dtype=np.float32),
        "ind_seq_seq": np.asarray(ind_seq_seq, dtype=np.float32),
        "ind_hor_seq": np.asarray(ind_hor_seq, dtype=np.float32),
        "time_hor": np.asarray(time_hor, dtype=np.float32),
        "target_seq": np.asarray(target_seq, dtype=np.float32),
    }


class TrafficDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        stats: Dict[str, np.ndarray],
        time_stats: Dict[str, int],
        node: np.ndarray,
        node_stats: Dict[str, np.ndarray],
    ) -> None:
        self.data = data
        self.stats = stats
        self.time_stats = time_stats
        self.node = node.astype(np.float32)
        self.node_stats = node_stats

        self.dyna_std = _safe_std(stats["dyna_traffic_std"])
        self.weather_std = _safe_std(stats["ext_weather_std"])
        self.target_std = _safe_std(stats["target_std"])
        self.ind_seq_range = np.maximum(stats["ind_seq_max"] - stats["ind_seq_min"], EPS)
        self.ind_hor_range = np.maximum(stats["ind_hor_max"] - stats["ind_hor_min"], EPS)
        self.node_std = _safe_std(node_stats["node_std"])

    def __len__(self) -> int:
        return int(self.data["target_seq"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dyna_traffic = (self.data["dyna_traffic"][idx] - self.stats["dyna_traffic_mean"]) / self.dyna_std
        ext_weather = (self.data["ext_weather_seq"][idx] - self.stats["ext_weather_mean"]) / self.weather_std
        target = (self.data["target_seq"][idx] - self.stats["target_mean"]) / self.target_std

        time_seq = self.data["time_seq"][idx][:, 0]
        time_hor = self.data["time_hor"][idx][:, 0]
        time_seq = self.encode_time(time_seq)
        time_hor = self.encode_time(time_hor)

        num_nodes = dyna_traffic.shape[1]
        time_seq = np.expand_dims(time_seq, axis=1).repeat(num_nodes, axis=1)
        time_hor = np.expand_dims(time_hor, axis=1).repeat(num_nodes, axis=1)

        ind_seq = (self.data["ind_seq_seq"][idx] - self.stats["ind_seq_min"]) / self.ind_seq_range
        ind_hor = (self.data["ind_hor_seq"][idx] - self.stats["ind_hor_min"]) / self.ind_hor_range
        node_features = (self.node - self.node_stats["node_mean"]) / self.node_std

        return {
            "dyna_traffic": torch.tensor(dyna_traffic, dtype=torch.float32),
            "ext_weather": torch.tensor(ext_weather, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "time_seq": torch.tensor(time_seq, dtype=torch.float32),
            "time_hor": torch.tensor(time_hor, dtype=torch.float32),
            "ind_seq": torch.tensor(ind_seq, dtype=torch.float32),
            "ind_hor": torch.tensor(ind_hor, dtype=torch.float32),
            "node": torch.tensor(node_features, dtype=torch.float32),
        }

    def encode_time(self, time_array: np.ndarray) -> np.ndarray:
        timestamps = pd.to_datetime(time_array.flatten(), unit="s", errors="coerce", utc=True)
        hour = timestamps.hour.to_numpy()
        weekday = timestamps.weekday.to_numpy()
        hour_sin = np.sin(2 * np.pi * hour / self.time_stats["hour_cycle"])
        hour_cos = np.cos(2 * np.pi * hour / self.time_stats["hour_cycle"])
        weekday_sin = np.sin(2 * np.pi * weekday / self.time_stats["weekday_cycle"])
        weekday_cos = np.cos(2 * np.pi * weekday / self.time_stats["weekday_cycle"])
        encoded = np.stack([hour_sin, hour_cos, weekday_sin, weekday_cos], axis=-1)
        return encoded.reshape(time_array.shape + (4,)).astype(np.float32)


def split_data_dict(
    data: Dict[str, np.ndarray],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    total = data["target_seq"].shape[0]
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = {key: value[:train_end] for key, value in data.items()}
    val_data = {key: value[train_end:val_end] for key, value in data.items()}
    test_data = {key: value[val_end:] for key, value in data.items()}
    return train_data, val_data, test_data


def build_data_bundle(
    dataset_dir: Path,
    sequence_length: int = 8,
    forecast_horizon: int = 4,
    batch_size: int = 16,
    num_workers: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
    test_ratio: float = 0.3,
) -> DataBundle:
    dataset_dir = Path(dataset_dir)
    geo, rel, dyna, ext, fut = read_streetsz_tables(dataset_dir)
    dyna_traffic, time_unix, ext_weather, ind_seq, ind_hor, adjacency_matrix, node_array = read_and_process(
        geo=geo,
        rel=rel,
        dyna=dyna,
        ext=ext,
        fut=fut,
    )
    data = create_sequences(
        dyna_traffic=dyna_traffic,
        time_unix=time_unix,
        ext_weather=ext_weather,
        ind_seq=ind_seq,
        ind_hor=ind_hor,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )

    stats = {
        "dyna_traffic_mean": data["dyna_traffic"].mean(axis=0).astype(np.float32),
        "dyna_traffic_std": data["dyna_traffic"].std(axis=0).astype(np.float32),
        "ext_weather_mean": data["ext_weather_seq"].mean(axis=0).astype(np.float32),
        "ext_weather_std": data["ext_weather_seq"].std(axis=0).astype(np.float32),
        "ind_seq_max": data["ind_seq_seq"].max(axis=0).astype(np.float32),
        "ind_seq_min": data["ind_seq_seq"].min(axis=0).astype(np.float32),
        "ind_hor_max": data["ind_hor_seq"].max(axis=0).astype(np.float32),
        "ind_hor_min": data["ind_hor_seq"].min(axis=0).astype(np.float32),
        "target_mean": data["target_seq"].mean(axis=0).astype(np.float32),
        "target_std": data["target_seq"].std(axis=0).astype(np.float32),
    }
    time_stats = {"hour_cycle": 24, "weekday_cycle": 7}
    node_stats = {
        "node_mean": node_array.mean(axis=0).astype(np.float32),
        "node_std": node_array.std(axis=0).astype(np.float32),
    }

    train_data, val_data, test_data = split_data_dict(
        data=data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_dataset = TrafficDataset(train_data, stats, time_stats, node_array, node_stats)
    val_dataset = TrafficDataset(val_data, stats, time_stats, node_array, node_stats)
    test_dataset = TrafficDataset(test_data, stats, time_stats, node_array, node_stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample = train_dataset[0]
    dims = {
        "node_feature_dim": int(sample["node"].shape[-1]),
        "var_dim_dyna": int(sample["dyna_traffic"].shape[-1]),
        "var_dim_weather": int(sample["ext_weather"].shape[-1]),
        "event_dim": int(sample["ind_seq"].shape[-1]),
        "time_dim": int(sample["time_seq"].shape[-1]),
        "target_dim": int(sample["target"].shape[-1]),
    }

    return DataBundle(
        dataset_dir=dataset_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adjacency_matrix=adjacency_matrix.astype(np.float32),
        stats=stats,
        node_stats=node_stats,
        dims=dims,
    )


def scenario_indices_from_dataset(dataset: TrafficDataset, scenario_name: str) -> List[int]:
    if scenario_name not in SCENARIO_CHANNELS:
        raise KeyError(f"Unknown scenario: {scenario_name}")
    channel = SCENARIO_CHANNELS[scenario_name]
    indicators = dataset.data["ind_hor_seq"][:, :, :, channel]
    mask = (indicators != 0).any(axis=(1, 2))
    return np.nonzero(mask)[0].astype(int).tolist()


def anomalous_indices_from_dataset(dataset: TrafficDataset) -> List[int]:
    union = set()
    for scenario_name in SCENARIO_CHANNELS:
        union.update(scenario_indices_from_dataset(dataset, scenario_name))
    return sorted(union)
