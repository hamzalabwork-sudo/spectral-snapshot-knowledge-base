from __future__ import annotations

"""
Data pipeline for SAPHIR-TS.

Supports:
    - Loading CSV time-series data
    - Sliding window creation
    - Train / val / test splitting
    - Continual learning regime simulation (sequential tasks)
    - Normalisation (per-feature StandardScaler)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


SEASONAL_REGIMES_ETTH = [
    ("2016-07-01", "2016-08-31"),   # R1: Summer 2016
    ("2016-09-01", "2017-02-28"),   # R2: Fall+Winter 2016
    ("2017-03-01", "2017-08-31"),   # R3: Spring+Summer 2017
    ("2017-09-01", "2018-02-28"),   # R4: Fall+Winter 2017
    ("2018-03-01", "2018-06-26"),   # R5: Spring 2018
]

# Weather: 10-min resolution. Use 5 contiguous seasonal-like blocks
# with sufficient length to produce valid train/test sliding windows.
SEASONAL_REGIMES_WEATHER = [
    ("2020-01-01", "2020-03-14"),   # R1: Winter to early Spring
    ("2020-03-15", "2020-05-27"),   # R2: Spring
    ("2020-05-28", "2020-08-09"),   # R3: Summer
    ("2020-08-10", "2020-10-22"),   # R4: Late Summer to Fall
    ("2020-10-23", "2021-01-01"),   # R5: Late Fall to early Winter
]

# Traffic: hourly, 2016-07 to 2018-07
SEASONAL_REGIMES_TRAFFIC = [
    ("2016-07-01", "2016-12-31"),   # R1: Summer+Fall 2016
    ("2017-01-01", "2017-06-30"),   # R2: Winter+Spring 2017
    ("2017-07-01", "2017-12-31"),   # R3: Summer+Fall 2017
    ("2018-01-01", "2018-04-30"),   # R4: Winter+Spring 2018
    ("2018-05-01", "2018-07-02"),   # R5: Summer 2018
]

# Electricity: hourly, 2016-07 to 2019-07
SEASONAL_REGIMES_ELECTRICITY = [
    ("2016-07-01", "2017-02-28"),   # R1: Summer+Fall+Winter 2016
    ("2017-03-01", "2017-10-31"),   # R2: Spring+Summer+Fall 2017
    ("2017-11-01", "2018-06-30"),   # R3: Winter+Spring 2018
    ("2018-07-01", "2019-02-28"),   # R4: Summer+Fall+Winter 2018
    ("2019-03-01", "2019-07-02"),   # R5: Spring+Summer 2019
]

# Registry for regime_split lookup
SEASONAL_REGIME_REGISTRY = {
    "seasonal": SEASONAL_REGIMES_ETTH,
    "seasonal_etth": SEASONAL_REGIMES_ETTH,
    "seasonal_weather": SEASONAL_REGIMES_WEATHER,
    "seasonal_traffic": SEASONAL_REGIMES_TRAFFIC,
    "seasonal_electricity": SEASONAL_REGIMES_ELECTRICITY,
}


def _build_seasonal_boundaries(dates: pd.Series, regime_defs: list) -> list[int]:
    """Convert date-range regime definitions into integer index boundaries."""
    dates = pd.to_datetime(dates)
    boundaries = [0]
    for _, end_date in regime_defs:
        mask = dates <= end_date
        boundaries.append(int(mask.sum()))
    # Make sure last boundary == total length
    boundaries[-1] = len(dates)
    return boundaries

# ===================================================================
# Normaliser
# ===================================================================
class StandardScaler:
    """Per-feature z-score normalisation."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        """Fit on (N, D) array."""
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.std = state["std"]


# ===================================================================
# Sliding Window Dataset
# ===================================================================
class TimeSeriesDataset(Dataset):
    """
    Creates (input_window, target_window) pairs from a contiguous
    time-series array using a sliding window.

    Args:
        data:     (T, D) numpy array
        seq_len:  L - input window length
        pred_len: H - prediction horizon
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        target_data: Optional[np.ndarray] = None,
    ):
        self.input_data = torch.tensor(data, dtype=torch.float32)
        if target_data is None:
            target_data = data
        self.target_data = torch.tensor(target_data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len

    def __len__(self) -> int:
        return max(0, len(self.input_data) - self.total_len + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_data[idx : idx + self.seq_len]                         # (L, D_in)
        y = self.target_data[idx + self.seq_len : idx + self.total_len]       # (H, D_out)
        return x, y


def _select_feature_target_arrays(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]],
    target_cols: Optional[list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Select feature/target matrices from a numeric dataframe."""
    if feature_cols is None:
        feature_df = df
    else:
        feature_df = df[feature_cols]

    if target_cols is None:
        target_df = feature_df
    else:
        target_df = df[target_cols]

    features = feature_df.values.astype(np.float32)
    targets = target_df.values.astype(np.float32)
    return features, targets


# ===================================================================
# Continual Regime Dataset
# ===================================================================
class ContinualRegimeDataset:
    """
    Splits a long time-series into sequential regimes (tasks) to
    simulate distribution shift in a continual learning setting.

    Usage:
        crd = ContinualRegimeDataset("data.csv", n_regimes=5, ...)
        for regime_id, loader in crd.regime_loaders():
            for x, y in loader:
                ...
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 96,
        pred_len: int = 24,
        n_regimes: int = 5,
        batch_size: int = 32,
        num_workers: int = 4,
        feature_cols: Optional[list[str]] = None,
        target_cols: Optional[list[str]] = None,
        date_col: str = "date",
        regime_split: str = "equal",          # <- NEW: "equal" or "seasonal"
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load data
        df = pd.read_csv(csv_path)

        # --- SEASONAL SPLIT: grab dates BEFORE dropping the column ---
        if regime_split in SEASONAL_REGIME_REGISTRY:
            if date_col not in df.columns:
                raise ValueError(f"regime_split='{regime_split}' requires '{date_col}' column")
            regime_defs = SEASONAL_REGIME_REGISTRY[regime_split]
            self.regime_boundaries = _build_seasonal_boundaries(
                df[date_col], regime_defs
            )
            self.n_regimes = len(regime_defs)
        else:
            self.n_regimes = n_regimes

        if date_col in df.columns:
            df = df.drop(columns=[date_col])

        features_raw, targets_raw = _select_feature_target_arrays(
            df, feature_cols=feature_cols, target_cols=target_cols
        )

        # Fit scalers on the full continual stream.
        self.feature_scaler = StandardScaler().fit(features_raw)
        self.target_scaler = StandardScaler().fit(targets_raw)
        self.features = self.feature_scaler.transform(features_raw)
        self.targets = self.target_scaler.transform(targets_raw)
        self.scaler = self.target_scaler

        # --- EQUAL SPLIT: only compute linspace if not seasonal ---
        if regime_split == "equal":
            T = len(self.features)
            self.regime_boundaries = np.linspace(0, T, self.n_regimes + 1, dtype=int)

    def regime_loaders(self):
        """Yield (regime_id, DataLoader) for each regime sequentially."""
        for i in range(self.n_regimes):
            start = self.regime_boundaries[i]
            end = self.regime_boundaries[i + 1]
            feature_chunk = self.features[start:end]
            target_chunk = self.targets[start:end]

            ds = TimeSeriesDataset(
                feature_chunk,
                self.seq_len,
                self.pred_len,
                target_data=target_chunk,
            )
            loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            yield i, loader


# ===================================================================
# Simple train/val/test split loader
# ===================================================================
def build_dataloaders(
    csv_path: str,
    seq_len: int = 96,
    pred_len: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    feature_cols: Optional[list[str]] = None,
    target_cols: Optional[list[str]] = None,
    date_col: str = "date",
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Build train / val / test DataLoaders from a CSV file.

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    df = pd.read_csv(csv_path)
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    features_raw, targets_raw = _select_feature_target_arrays(
        df, feature_cols=feature_cols, target_cols=target_cols
    )
    T = len(features_raw)

    # Split points
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_features_raw = features_raw[:train_end]
    val_features_raw = features_raw[train_end:val_end]
    test_features_raw = features_raw[val_end:]

    train_targets_raw = targets_raw[:train_end]
    val_targets_raw = targets_raw[train_end:val_end]
    test_targets_raw = targets_raw[val_end:]

    # Fit scalers on training split only.
    feature_scaler = StandardScaler().fit(train_features_raw)
    target_scaler = StandardScaler().fit(train_targets_raw)

    train_features = feature_scaler.transform(train_features_raw)
    val_features = feature_scaler.transform(val_features_raw)
    test_features = feature_scaler.transform(test_features_raw)

    train_targets = target_scaler.transform(train_targets_raw)
    val_targets = target_scaler.transform(val_targets_raw)
    test_targets = target_scaler.transform(test_targets_raw)

    train_ds = TimeSeriesDataset(
        train_features, seq_len, pred_len, target_data=train_targets
    )
    val_ds = TimeSeriesDataset(
        val_features, seq_len, pred_len, target_data=val_targets
    )
    test_ds = TimeSeriesDataset(
        test_features, seq_len, pred_len, target_data=test_targets
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, target_scaler
