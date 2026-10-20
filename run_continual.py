from __future__ import annotations

"""
SpectraCL / MambaCL Continual Learning Experiments.

Runs sequential methods on temporal tasks and measures forgetting:
    1. finetune  -- naive sequential training (no protection)
    2. ewc       -- Elastic Weight Consolidation
    3. si        -- Synaptic Intelligence
    4. mas       -- Memory Aware Synapses
    5. replay    -- experience replay buffer
    6. derpp     -- replay + distillation
    7. fsnet     -- Fast and Slow Network
    8. spectracl -- SpectraCL (our method: spectral replay + null-space + state partition)
    9. joint     -- train on all data (upper bound)
    10. adaptivecl -- Wu et al. adaptive dual-memory + hint regularisation baseline
    11. clser -- CLS-ER style dual-EMA replay consistency baseline

Usage:
    python run_continual.py --config configs/continual_etth1_h24.yaml --data data/ETTh1.csv
    python run_continual.py --config configs/continual_etth1_h24.yaml --data data/ETTh1.csv --methods finetune replay spectracl
"""

import os
import sys
import argparse
import random
import time
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectracl.model import MambaForecaster
from spectracl.continual.ewc import EWC
from spectracl.continual.replay import ReplayBuffer
from spectracl.continual.si import SI
from spectracl.continual.mas import MAS
from spectracl.continual.derpp import DERPPBuffer
from spectracl.continual.fsnet import FSNetAdapter
from spectracl.continual.spectral_replay import SpectralReplayBuffer
from spectracl.continual.spectral_regularization import SpectralRegularizer
from spectracl.continual.null_space import NullSpaceProjector
from spectracl.continual.state_partition import StatePartitioner
from spectracl.continual.hint_loss import FrequencyHintLoss
from spectracl.continual.spectral_gradient import SpectralGradientDecomposer
from spectracl.continual.adaptivecl import AdaptiveCLBuffer
from spectracl.continual.clser import CLSERAdapter
from spectracl.continual.intrusion_rehearsal import IntrusionRehearsal, SpectralWeighter, fredf_loss
from spectracl.clfd import CLFDForecaster
from spectracl.spectral_gate import SpectralGate
from spectracl.linear import LinearForecaster
from spectracl.routing import SnapshotRouter

# Candidate methods with the input-conditioned spectral output gate.
GATED_METHODS = ("sg_ft", "sg_er", "sg_ifr", "clfd_sg_ifr")
from spectracl.continual.mir import MIRReplay
from spectracl.continual.agem import AGEM

# Ablations of IFR: method name -> weighting scheme (ifr_uniform uses kappa = 0, i.e. ER).
IFR_ABLATIONS = {
    "ifr_uniform": "intrusion",
    "ifr_old_energy": "old_energy",
    "ifr_overlap": "overlap",
    "ifr_neglect": "neglect",
    "ifr_shuffled": "shuffled",
    "ifr_pooled": "pooled",
}
from data.dataloader import (
    StandardScaler,
    TimeSeriesDataset,
    SEASONAL_REGIME_REGISTRY,
    _build_seasonal_boundaries,
)

import pandas as pd
from torch.utils.data import DataLoader


# ===================================================================
# Helpers
# ===================================================================
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stored_state_bytes(objects: list, exclude: nn.Module) -> int:
    """Bytes of tensors a CL method keeps besides the trained model.

    Walks the method's helper objects (buffers, Fisher/importance tensors, EMA or frozen
    model copies, spectra, ...) and sums tensor storage, counting each tensor once and
    skipping the live model's own parameters and buffers.
    """
    skip = {id(exclude)} | {id(t) for t in exclude.parameters()} | {id(t) for t in exclude.buffers()}
    seen: set[int] = set()

    def walk(o) -> int:
        if o is None or id(o) in skip or id(o) in seen:
            return 0
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            return o.numel() * o.element_size()
        if isinstance(o, nn.Module):
            return sum(walk(t) for t in list(o.parameters()) + list(o.buffers()))
        if isinstance(o, dict):
            return sum(walk(v) for v in o.values())
        if isinstance(o, (list, tuple, set)):
            return sum(walk(v) for v in o)
        if hasattr(o, "__dict__") and not isinstance(o, (DataLoader, torch.optim.Optimizer)):
            caches = set(getattr(o, "_cache_attrs", ()))   # speed-only copies of stored data
            return sum(walk(v) for k, v in vars(o).items() if k not in caches)
        return 0

    return sum(walk(o) for o in objects)


@torch.no_grad()
def inference_latency_ms(model: nn.Module, loader: DataLoader, device: str, repeats: int = 50) -> float:
    """Mean forward time (ms) for one test batch, after warm-up."""
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)
    for _ in range(10):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / repeats


# ===================================================================
# Data: Build task-specific train/test loaders
# ===================================================================
def _clean_sentinels(df: pd.DataFrame, sentinel: float = -9999.0) -> pd.DataFrame:
    """Replace missing-value sentinels (e.g. -9999 in Weather) by linear interpolation."""
    num = df.select_dtypes(include="number").columns
    bad = df[num] <= sentinel
    if bad.values.any():
        print(f"Cleaning {int(bad.values.sum())} sentinel values (<= {sentinel}) by interpolation")
        df[num] = df[num].mask(bad).interpolate(limit_direction="both")
    return df


def resolve_target_indices(
    csv_path: str,
    feature_cols: list[str] | None,
    target_cols: list[str] | None,
    date_col: str = "date",
) -> list[int] | None:
    """Positions of the target columns among the input features.

    Needed so RevIN can de-normalise predictions when the model forecasts a
    subset of its input channels (e.g. all features -> OT). Returns None when
    inputs and targets are the same channels.
    """
    if target_cols is None:
        return None
    cols = [c for c in pd.read_csv(csv_path, nrows=0).columns if c != date_col]
    feats = feature_cols if feature_cols is not None else cols
    missing = [t for t in target_cols if t not in feats]
    if missing:
        raise ValueError(f"Target columns {missing} are not among the input features")
    idx = [feats.index(t) for t in target_cols]
    return None if idx == list(range(len(feats))) else idx


def _compute_task_boundaries(
    df: pd.DataFrame,
    n_tasks: int,
    task_split: str,
    date_col: str,
) -> np.ndarray:
    if task_split in SEASONAL_REGIME_REGISTRY:
        if date_col not in df.columns:
            raise ValueError(f"task_split='{task_split}' requires '{date_col}' column")
        regime_defs = SEASONAL_REGIME_REGISTRY[task_split]
        boundaries = _build_seasonal_boundaries(df[date_col], regime_defs)
        return np.asarray(boundaries, dtype=int)
    T = len(df)
    return np.linspace(0, T, n_tasks + 1, dtype=int)


MIN_CASE_WINDOWS = 32


def _case_length(train_len: int, case_ratio: float, total_window: int) -> int:
    """Points held out as the case segment: case_ratio of the segment, but >= MIN_CASE_WINDOWS windows."""
    return max(int(train_len * case_ratio), total_window + MIN_CASE_WINDOWS - 1)


def build_task_data(
    csv_path: str,
    seq_len: int,
    pred_len: int,
    n_tasks: int,
    batch_size: int,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    date_col: str = "date",
    task_test_ratio: float = 0.2,
    task_split: str = "equal",
    holdout: str = "test",
    val_ratio: float = 0.2,
    case_ratio: float = 0.0,
    case_sets: list | None = None,
) -> tuple[list[DataLoader], list[DataLoader], StandardScaler]:
    """Per-task chronological train/eval loaders.

    holdout="test": train on the first (1 - task_test_ratio) of each task, evaluate on the rest.
    holdout="val":  the test segment is discarded; the training segment is split again
                    chronologically and its last `val_ratio` is used for evaluation. Use this
                    for all hyperparameter selection so test data is never seen.
    case_ratio > 0: the last `case_ratio` of each training segment (at least MIN_CASE_WINDOWS
                    windows) is held out from training and appended to `case_sets` as a dataset
                    (case base for snapshot routing).
    """
    if holdout not in ("test", "val"):
        raise ValueError(f"holdout must be 'test' or 'val', got {holdout!r}")
    df = _clean_sentinels(pd.read_csv(csv_path))
    boundaries = _compute_task_boundaries(df, n_tasks, task_split, date_col)
    n_tasks = len(boundaries) - 1

    if n_tasks <= 0:
        raise ValueError(
            f"No tasks were created for split '{task_split}'. Boundaries: {boundaries.tolist()}"
        )

    if np.any(np.diff(boundaries) < 0):
        raise ValueError(
            f"Non-monotonic task boundaries for split '{task_split}': {boundaries.tolist()}"
        )

    total_window = seq_len + pred_len
    task_debug = []
    bad_tasks = []
    for i in range(n_tasks):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        regime_len = end - start
        split = start + int(regime_len * (1 - task_test_ratio))
        train_len = split - start
        test_len = end - split
        train_windows = max(0, train_len - total_window + 1)
        test_windows = max(0, test_len - total_window + 1)
        task_debug.append(
            f"T{i}: regime={regime_len}, train={train_len}, test={test_len}, "
            f"train_windows={train_windows}, test_windows={test_windows}"
        )
        if train_windows <= 0:
            bad_tasks.append(i)

    if bad_tasks:
        raise ValueError(
            "One or more tasks have zero train windows. "
            f"task_split='{task_split}', seq_len={seq_len}, pred_len={pred_len}, "
            f"boundaries={boundaries.tolist()}, bad_tasks={bad_tasks}. "
            "This usually means split/date ranges do not match the dataset (e.g. ETTh1 with "
            "seasonal_electricity) or windows are too long for a regime. "
            "Per-task stats: " + " | ".join(task_debug)
        )

    if date_col in df.columns:
        df = df.drop(columns=[date_col])

    feature_df = df[feature_cols] if feature_cols is not None else df
    target_df = df[target_cols] if target_cols is not None else feature_df

    features_raw = feature_df.values.astype(np.float32)
    targets_raw = target_df.values.astype(np.float32)

    r0_end = boundaries[1]
    r0_train_end = int(r0_end * (1 - task_test_ratio))
    if r0_train_end <= 0:
        raise ValueError(
            f"Task-0 train split is empty: r0_end={r0_end}, task_test_ratio={task_test_ratio}."
        )
    feature_scaler = StandardScaler().fit(features_raw[:r0_train_end])
    target_scaler = StandardScaler().fit(targets_raw[:r0_train_end])

    features = feature_scaler.transform(features_raw)
    targets = target_scaler.transform(targets_raw)

    train_loaders = []
    test_loaders = []

    for i in range(n_tasks):
        start = boundaries[i]
        end = boundaries[i + 1]
        regime_len = end - start
        split = start + int(regime_len * (1 - task_test_ratio))
        if holdout == "val":
            end = split
            split = start + int((split - start) * (1 - val_ratio))

        train_end = split
        if case_ratio > 0:
            train_end = split - _case_length(split - start, case_ratio, total_window)
            case_sets.append(TimeSeriesDataset(features[train_end:split], seq_len, pred_len,
                                               target_data=targets[train_end:split]))

        train_feats = features[start:train_end]
        train_tgts = targets[start:train_end]
        test_feats = features[split:end]
        test_tgts = targets[split:end]

        train_ds = TimeSeriesDataset(train_feats, seq_len, pred_len, target_data=train_tgts)
        test_ds = TimeSeriesDataset(test_feats, seq_len, pred_len, target_data=test_tgts)

        train_loaders.append(DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True,
        ))
        test_loaders.append(DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        ))

    return train_loaders, test_loaders, target_scaler


def build_multidomain_task_data(
    domains: list[dict],
    seq_len: int,
    pred_len: int,
    batch_size: int,
    task_test_ratio: float = 0.2,
    holdout: str = "test",
    val_ratio: float = 0.2,
    data_root: str = ".",
    case_ratio: float = 0.0,
    case_sets: list | None = None,
) -> tuple[list[DataLoader], list[DataLoader], None]:
    """Cross-domain continual forecasting: each task is a univariate series from a different source.

    Each entry of `domains` is {"csv": path, "column": name, "length": N}. The first N
    points of the column are used (sentinels cleaned), split chronologically into
    train / evaluation exactly as in build_task_data, and standardised with the
    task's own training statistics (available when the task arrives).
    """
    if holdout not in ("test", "val"):
        raise ValueError(f"holdout must be 'test' or 'val', got {holdout!r}")
    train_loaders, test_loaders = [], []
    for d in domains:
        df = _clean_sentinels(pd.read_csv(os.path.join(data_root, d["csv"]), usecols=[d["column"]]))
        series = df[d["column"]].values.astype(np.float32)[: d.get("length", None)].reshape(-1, 1)
        split = int(len(series) * (1 - task_test_ratio))
        end = len(series)
        if holdout == "val":
            end = split
            split = int(split * (1 - val_ratio))
        scaler = StandardScaler().fit(series[:split])
        series = scaler.transform(series)
        train_end = split
        if case_ratio > 0:   # held-out case segment, as in build_task_data
            train_end = split - _case_length(split, case_ratio, seq_len + pred_len)
            case_sets.append(TimeSeriesDataset(series[train_end:split], seq_len, pred_len))
        train_ds = TimeSeriesDataset(series[:train_end], seq_len, pred_len)
        test_ds = TimeSeriesDataset(series[split:end], seq_len, pred_len)
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                        num_workers=2, pin_memory=True, drop_last=True))
        test_loaders.append(DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                       num_workers=2, pin_memory=True))
    return train_loaders, test_loaders, None


def build_joint_loader(
    csv_path: str, seq_len: int, pred_len: int, n_tasks: int,
    batch_size: int, feature_cols=None, target_cols=None, date_col="date",
    task_test_ratio: float = 0.2, task_split: str = "equal",
) -> DataLoader:
    df = _clean_sentinels(pd.read_csv(csv_path))
    boundaries = _compute_task_boundaries(df, n_tasks, task_split, date_col)
    n_tasks = len(boundaries) - 1

    if date_col in df.columns:
        df = df.drop(columns=[date_col])

    feature_df = df[feature_cols] if feature_cols is not None else df
    target_df = df[target_cols] if target_cols is not None else feature_df

    features_raw = feature_df.values.astype(np.float32)
    targets_raw = target_df.values.astype(np.float32)

    r0_end = boundaries[1]
    r0_train_end = int(r0_end * (1 - task_test_ratio))
    feature_scaler = StandardScaler().fit(features_raw[:r0_train_end])
    target_scaler = StandardScaler().fit(targets_raw[:r0_train_end])

    features = feature_scaler.transform(features_raw)
    targets = target_scaler.transform(targets_raw)

    all_train_f, all_train_t = [], []
    for i in range(n_tasks):
        start = boundaries[i]
        end = boundaries[i + 1]
        split = start + int((end - start) * (1 - task_test_ratio))
        all_train_f.append(features[start:split])
        all_train_t.append(targets[start:split])

    all_f = np.concatenate(all_train_f)
    all_t = np.concatenate(all_train_t)
    ds = TimeSeriesDataset(all_f, seq_len, pred_len, target_data=all_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                      pin_memory=True, drop_last=True)


# ===================================================================
# Create fresh model
# ===================================================================
def create_model(cfg: dict, num_features: int, output_features: int, device: str,
                 method: str = "") -> MambaForecaster:
    bc = cfg["backbone"]
    dc = cfg["data"]
    if cfg.get("model", {}).get("backbone") == "linear":
        return LinearForecaster(
            num_features=num_features, output_features=output_features,
            seq_len=dc["seq_len"], pred_len=dc["pred_len"],
            use_revin=cfg.get("model", {}).get("use_revin", True),
            target_indices=dc.get("target_indices"),
        ).to(device)
    model_cls = CLFDForecaster if method.startswith("clfd") else MambaForecaster
    model = model_cls(
        num_features=num_features,
        output_features=output_features,
        seq_len=dc["seq_len"],
        pred_len=dc["pred_len"],
        d_model=bc["d_model"],
        d_state=bc["d_state"],
        d_conv=bc["d_conv"],
        expand=bc["expand"],
        n_layers=bc["n_layers"],
        dropout=bc["dropout"],
        use_revin=cfg.get("model", {}).get("use_revin", True),
        target_indices=dc.get("target_indices"),
    )
    if method in GATED_METHODS:
        model.spectral_gate = SpectralGate(dc["seq_len"], dc["pred_len"])
    return model.to(device)


# ===================================================================
# Train one task
# ===================================================================
def train_task(
    model: MambaForecaster,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    grad_clip: float = 1.0,
    ewc: EWC | None = None,
    si: SI | None = None,
    mas: MAS | None = None,
    replay_loader: DataLoader | None = None,
    derpp: DERPPBuffer | None = None,
    fsnet_adapter=None,
    # SpectraCL components
    spectral_buffer: SpectralReplayBuffer | None = None,
    anchor_replay: ReplayBuffer | None = None,
    anchor_replay_weight: float = 1.0,
    null_projector: NullSpaceProjector | None = None,
    state_partitioner: StatePartitioner | None = None,
    spectral_reg: SpectralRegularizer | None = None,
    sfr_interval: int = 3,
    replay_steps: int = 0,
    # Novel SpectraCL components
    hint_loss: FrequencyHintLoss | None = None,
    hint_interval: int = 3,
    sgbd: SpectralGradientDecomposer | None = None,
    # Wu et al. baseline components
    adaptive_cl: AdaptiveCLBuffer | None = None,
    # CLS-ER baseline components
    clser: CLSERAdapter | None = None,
    # Intrusion-aware frequency rehearsal
    ifr: IntrusionRehearsal | None = None,
    # IFR plug-in: re-weights the replay losses of DER++ / CLS-ER
    spectral_weighter: SpectralWeighter | None = None,
    # Additional comparison baselines
    mir: MIRReplay | None = None,
    agem: AGEM | None = None,
    verbose: bool = True,
) -> None:
    """Train model on a single task with optional CL mechanisms."""

    if si is not None:
        si.begin_regime(model)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # Get input dimensions for SFR probes
        sfr_dims = None
        if spectral_reg is not None and spectral_reg.is_active:
            sample_x, _ = next(iter(train_loader))
            sfr_dims = (sample_x.shape[1], sample_x.shape[2])

        step_counter = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Wu et al. adaptive CL: combine current batch with soft-memory samples.
            x_all, y_all = x, y
            if adaptive_cl is not None:
                soft_batch = adaptive_cl.sample_soft()
                if soft_batch is not None:
                    x_soft, y_soft = soft_batch
                    x_all = torch.cat([x, x_soft.to(device)], dim=0)
                    y_all = torch.cat([y, y_soft.to(device)], dim=0)

            use_weighter = spectral_weighter is not None and spectral_weighter.is_active

            # CLS-ER: combine replay samples into supervised batch.
            clser_tids = None
            if clser is not None and clser.buffer.size > 0:
                clser_batch = clser.buffer.sample(x.shape[0], return_task_ids=use_weighter)
                if clser_batch is not None:
                    x_replay, y_replay = clser_batch[0], clser_batch[1]
                    if use_weighter:
                        clser_tids = clser_batch[2].to(device)
                    x_all = torch.cat([x_all, x_replay.to(device)], dim=0)
                    y_all = torch.cat([y_all, y_replay.to(device)], dim=0)

            out = model(x_all, update_memory=False)
            if clser_tids is not None:
                # Same element weighting as the concatenated MSE, but the replay part
                # uses the intrusion-weighted frequency-domain loss.
                n_cur, n_rep = x.shape[0], clser_tids.shape[0]
                loss_cur = F.mse_loss(out["y_pred"][:n_cur], y_all[:n_cur])
                loss_rep = spectral_weighter.loss(out["y_pred"][n_cur:], y_all[n_cur:], clser_tids)
                loss = (n_cur * loss_cur + n_rep * loss_rep) / (n_cur + n_rep)
            elif ifr is not None and ifr.fredf_alpha is not None:
                loss = fredf_loss(out["y_pred"], y_all, ifr.fredf_alpha)
            else:
                loss = F.mse_loss(out["y_pred"], y_all)

            # Wu et al. adaptive CL regularisation terms (main memory + hint).
            if adaptive_cl is not None:
                mem_reg = adaptive_cl.main_regularization(model, device)
                if mem_reg is not None:
                    loss = loss + mem_reg

                hint_target = adaptive_cl.compute_hint_target(x, device)
                if hint_target is not None:
                    current_h = out["h"][: x.shape[0]]
                    loss = loss + adaptive_cl.gamma * F.mse_loss(current_h, hint_target)

            # CLS-ER: consistency loss against selected EMA teacher.
            if clser is not None:
                clser_cons = clser.consistency_loss(model, x.shape[0], device,
                                                    weighter=spectral_weighter)
                if clser_cons is not None:
                    loss = loss + clser_cons

            # EWC penalty
            if ewc is not None and ewc.is_registered:
                loss = loss + ewc.penalty(model)

            # SI penalty
            if si is not None and si.is_registered:
                loss = loss + si.penalty(model)

            # MAS penalty
            if mas is not None and mas.is_registered:
                loss = loss + mas.penalty(model)

            # DER++ replay
            if derpp is not None:
                replay_batch = derpp.sample(x.shape[0], return_task_ids=use_weighter)
                if replay_batch is not None:
                    xr, yr, y_teacher = (t.to(device) for t in replay_batch[:3])
                    out_r = model(xr, update_memory=False)
                    if use_weighter:
                        tids = replay_batch[3].to(device)
                        loss = loss + derpp.alpha * spectral_weighter.loss(out_r["y_pred"], y_teacher, tids)
                        loss = loss + derpp.beta * spectral_weighter.loss(out_r["y_pred"], yr, tids)
                    else:
                        loss = loss + derpp.alpha * F.mse_loss(out_r["y_pred"], y_teacher)
                        loss = loss + derpp.beta * F.mse_loss(out_r["y_pred"], yr)

            # SpectraCL: SIAR sampling + FDRL loss
            if spectral_buffer is not None and spectral_buffer.size > 0:
                replay_batch = spectral_buffer.sample_siar(x.shape[0], x, device)
                if replay_batch is not None:
                    xr, yr = replay_batch
                    out_r = model(xr, update_memory=False)
                    siar_mse_weight = getattr(spectral_buffer, "siar_mse_weight", 0.0)
                    if siar_mse_weight > 0:
                        loss = loss + siar_mse_weight * F.mse_loss(out_r["y_pred"], yr)
                    loss = loss + spectral_buffer.compute_fdrl(out_r["y_pred"], yr, device)

            # MIR: replay the buffer samples most interfered with by a virtual step on this batch
            if mir is not None:
                mir_loss = mir.replay_loss(model, loss, x.shape[0], device)
                if mir_loss is not None:
                    loss = loss + mir_loss

            # Intrusion-aware frequency rehearsal
            if ifr is not None:
                ifr_loss = ifr.replay_loss(model, x.shape[0])
                if ifr_loss is not None:
                    loss = loss + ifr_loss

            # SpectraCL anchor replay: direct time-domain MSE on raw stored samples.
            if (
                anchor_replay is not None
                and anchor_replay.size > 0
                and anchor_replay_weight > 0
            ):
                anchor_batch = anchor_replay.sample(x.shape[0])
                if anchor_batch is not None:
                    xa, ya = anchor_batch
                    xa, ya = xa.to(device), ya.to(device)
                    out_a = model(xa, update_memory=False)
                    loss = loss + anchor_replay_weight * F.mse_loss(out_a["y_pred"], ya)

            # SpectraCL: Spectral Frequency-Response Regularisation (SFR)
            if spectral_reg is not None and spectral_reg.is_active and sfr_dims is not None:
                if step_counter % sfr_interval == 0:
                    sfr_loss = spectral_reg.compute_loss(
                        model, sfr_dims[0], sfr_dims[1], device
                    )
                    if sfr_loss is not None:
                        loss = loss + sfr_loss

            # SpectraCL: Frequency-Selective Hint Loss (FSHL)
            if hint_loss is not None and hint_loss.is_active:
                if step_counter % hint_interval == 0:
                    loss = loss + hint_loss.compute_loss(model, x, device)

            step_counter += 1

            optimizer.zero_grad()
            loss.backward()

            # SpectraCL: Spectral Gradient Band Decomposition (SGBD)
            if sgbd is not None and sgbd.is_active:
                sgbd.scale_gradients(model)

            # SpectraCL: null-space gradient projection (AFTER backward, BEFORE step)
            if null_projector is not None and null_projector.is_active:
                null_projector.project_gradients(model)

            # SpectraCL: state partition gradient masking on A_log
            if state_partitioner is not None and state_partitioner.is_active:
                state_partitioner.apply_gradient_mask(model)

            # A-GEM: project the gradient if it would increase the loss on the memory
            if agem is not None:
                agem.project_gradients(model, x.shape[0], device)

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if fsnet_adapter is not None:
                fsnet_adapter.step(model, optimizer)
            else:
                optimizer.step()

            if clser is not None:
                clser.update_ema_models(model)

            if si is not None:
                si.update_after_step(model)

            # Wu et al. adaptive CL: update dual memory buffers after each step.
            if adaptive_cl is not None:
                with torch.no_grad():
                    y_pred_cur = out["y_pred"][: x.shape[0]].detach()
                    h_cur = out["h"][: x.shape[0]].detach()
                    per_sample_err = ((y_pred_cur - y) ** 2).mean(dim=(1, 2))

                adaptive_cl.add_main_batch(
                    x.detach().cpu(),
                    y.detach().cpu(),
                    h_cur.cpu(),
                )
                adaptive_cl.add_soft_batch(
                    x.detach().cpu(),
                    y.detach().cpu(),
                    per_sample_err.cpu(),
                )

            total_loss += loss.item()
            n_batches += 1

        # Standard replay: also train on old data (Replay baseline only)
        if replay_loader is not None:
            for x, y in replay_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, update_memory=False)
                loss = F.mse_loss(out["y_pred"], y)

                if ewc is not None and ewc.is_registered:
                    loss = loss + ewc.penalty(model)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if si is not None:
                    si.update_after_step(model)

        if verbose and (epoch % 5 == 0 or epoch == epochs):
            avg = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch:3d}/{epochs}  Loss: {avg:.6f}")

    if si is not None:
        si.end_regime(model)


# ===================================================================
# Evaluate on all tasks
# ===================================================================
@torch.no_grad()
def evaluate_all_tasks(
    model: MambaForecaster,
    test_loaders: list[DataLoader],
    device: str,
) -> list[dict[str, float]]:
    model.eval()
    results = []
    for loader in test_loaders:
        total_mse = 0.0
        total_mae = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, update_memory=False)
            total_mse += F.mse_loss(out["y_pred"], y).item()
            total_mae += F.l1_loss(out["y_pred"], y).item()
            n_batches += 1
        if n_batches > 0:
            results.append({"mse": total_mse / n_batches, "mae": total_mae / n_batches})
        else:
            results.append({"mse": float("nan"), "mae": float("nan")})
    return results


@torch.no_grad()
def collect_predictions(
    model: MambaForecaster,
    test_loaders: list[DataLoader],
    device: str,
) -> dict:
    """Collect raw predictions and ground truth for plotting."""
    model.eval()
    all_preds = {}
    for task_id, loader in enumerate(test_loaders):
        preds, truths = [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, update_memory=False)
            preds.append(out["y_pred"].cpu())
            truths.append(y.cpu())
        all_preds[task_id] = {
            "y_pred": torch.cat(preds, dim=0).numpy().tolist(),
            "y_true": torch.cat(truths, dim=0).numpy().tolist(),
        }
    return all_preds


# ===================================================================
# Run one continual learning method
# ===================================================================
def run_method(
    method: str,
    cfg: dict,
    train_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    joint_loader: DataLoader | None,
    device: str,
    num_features: int,
    output_features: int,
    case_sets: list | None = None,
    dump_path: str | None = None,
) -> dict:
    n_tasks = len(train_loaders)
    tc = cfg["training"]
    epochs_per_task = tc.get("epochs_per_regime", tc.get("epochs_per_task", 30))

    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}")
    print(f"{'='*60}")

    # --- Joint training (special case) ---
    if method == "joint":
        model = create_model(cfg, num_features, output_features, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                                      weight_decay=tc["weight_decay"])
        total_epochs = epochs_per_task * n_tasks
        print(f"Training on ALL tasks jointly for {total_epochs} epochs...")
        for epoch in range(1, total_epochs + 1):
            model.train()
            total_loss, n_batches = 0.0, 0
            for x, y in joint_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, update_memory=False)
                loss = F.mse_loss(out["y_pred"], y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d}  Loss: {total_loss / max(n_batches, 1):.6f}")

        final = evaluate_all_tasks(model, test_loaders, device)
        return {"method": method, "results_matrix": [final], "final_results": final}

    # --- Sequential methods ---
    model = create_model(cfg, num_features, output_features, device, method=method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                                  weight_decay=tc["weight_decay"])

    # Initialize method-specific components
    ewc = EWC(lambda_ewc=cfg.get("ewc", {}).get("lambda_ewc", 1000.0)) if method == "ewc" else None
    si = SI(lambda_si=cfg.get("si", {}).get("lambda_si", 1.0),
            epsilon=cfg.get("si", {}).get("epsilon", 0.1)) if method == "si" else None
    mas = MAS(lambda_mas=cfg.get("mas", {}).get("lambda_mas", 1.0)) if method == "mas" else None
    replay = ReplayBuffer(capacity=cfg.get("replay", {}).get("capacity", 500)) if method == "replay" else None
    derpp = DERPPBuffer(
        capacity=cfg.get("derpp", {}).get("capacity", 500),
        alpha=cfg.get("derpp", {}).get("alpha", 0.5),
        beta=cfg.get("derpp", {}).get("beta", 0.5),
    ) if method in ("derpp", "derpp_ifr") else None
    # IFR as a plug-in for existing replay methods (same hyperparameters as standalone IFR)
    spectral_weighter = SpectralWeighter(
        kappa=cfg.get("ifr", {}).get("kappa", 4.0),
    ) if method in ("derpp_ifr", "clser_ifr") else None
    fsnet_adapter = FSNetAdapter(
        model=model,
        gamma=cfg.get("fsnet", {}).get("gamma", 0.9),
        gamma_prime=cfg.get("fsnet", {}).get("gamma_prime", 0.99),
        memory_capacity=cfg.get("fsnet", {}).get("memory_capacity", 32),
        trigger_threshold=cfg.get("fsnet", {}).get("trigger_threshold", 0.1),
        adapt_lr=cfg.get("fsnet", {}).get("adapt_lr", 0.01),
    ) if method == "fsnet" else None

    # SpectraCL components
    spectral_buffer = None
    anchor_replay = None
    anchor_replay_weight = 1.0
    null_projector = None
    state_partitioner = None
    spectracl_ewc = None
    spectral_reg = None
    sfr_interval = 3
    hint_loss_obj = None
    hint_interval = 3
    sgbd_obj = None
    adaptive_cl = None
    clser_obj = None
    ifr_obj = None
    mir_obj = None
    agem_obj = None

    if method in ("fredf", "clfd"):
        # Both are rehearsal-based in our comparison: ER's buffer and sampling, plus
        # the FreDF objective (fredf) or CLFD's wavelet feature encoder (clfd).
        ic = cfg.get("ifr", {})
        ifr_obj = IntrusionRehearsal(
            capacity=ic.get("capacity", 500), kappa=0.0, weight=ic.get("weight", 1.0),
            fredf_alpha=cfg.get("fredf", {}).get("alpha", 0.8) if method == "fredf" else None,
        )
        print(f"    {method.upper()}: ER buffer capacity={ifr_obj.capacity}"
              + (f", FreDF alpha={ifr_obj.fredf_alpha}" if method == "fredf" else ", Haar FFE"))

    if method in ("ifr", "clfd_ifr", "sg_er", "sg_ifr", "clfd_sg_ifr") or method in IFR_ABLATIONS:
        ic = cfg.get("ifr", {})
        ifr_obj = IntrusionRehearsal(
            capacity=ic.get("capacity", 500),
            kappa=0.0 if method in ("ifr_uniform", "sg_er") else ic.get("kappa", 4.0),
            weight=ic.get("weight", 1.0),
            weighting=IFR_ABLATIONS.get(method, "intrusion"),
            # Drawn from the per-run RNG (seeded before each method), so the shuffled
            # control uses a different permutation for every seed.
            seed=int(torch.randint(0, 2**31 - 1, (1,)).item()),
        )
        print(f"    IFR: capacity={ifr_obj.capacity}, kappa={ifr_obj.kappa}, weight={ifr_obj.weight}, "
              f"weighting={ifr_obj.weighter.weighting}")

    if method == "mir":
        mc = cfg.get("mir", {})
        mir_obj = MIRReplay(
            capacity=mc.get("capacity", cfg.get("ifr", {}).get("capacity", 500)),
            n_candidates=mc.get("n_candidates", 5 * tc["batch_size"]),
            virtual_lr=tc["learning_rate"],
        )
        print(f"    MIR: capacity={mir_obj.buffer.capacity}, candidates={mir_obj.n_candidates}")

    if method == "agem":
        agem_obj = AGEM(capacity=cfg.get("agem", {}).get("capacity", cfg.get("ifr", {}).get("capacity", 500)))
        print(f"    A-GEM: capacity={agem_obj.buffer.capacity}")

    if method == "spectracl":
        sc = cfg.get("spectracl", {})
        dc = cfg["data"]
        siar_mse_weight = sc.get("siar_mse_weight", 0.0)
        use_csd = sc.get("use_csd", False)
        spectral_buffer_kwargs = {
            "capacity": sc.get("capacity", 500),
            "n_fft_fingerprint": sc.get("n_fft_fingerprint", 48),
            "fdrl_weight": sc.get("fdrl_weight", 1.0),
            "siar_temperature": sc.get("siar_temperature", 0.5),
            "siar_mse_weight": siar_mse_weight,
            "use_siar": sc.get("use_siar", True),
            "use_smc": sc.get("use_smc", True),
            "use_csd": use_csd,
            "csd_blend": sc.get("csd_blend", 0.3),
        }
        try:
            spectral_buffer = SpectralReplayBuffer(**spectral_buffer_kwargs)
        except TypeError:
            # Backward compatibility with older SpectralReplayBuffer versions.
            spectral_buffer_kwargs.pop("siar_mse_weight", None)
            spectral_buffer_kwargs.pop("use_siar", None)
            spectral_buffer_kwargs.pop("use_smc", None)
            spectral_buffer_kwargs.pop("use_csd", None)
            spectral_buffer_kwargs.pop("csd_blend", None)
            spectral_buffer = SpectralReplayBuffer(**spectral_buffer_kwargs)
            spectral_buffer.siar_mse_weight = siar_mse_weight
            # Older buffer versions do not implement CSD.
            spectral_buffer.use_csd = False
            spectral_buffer.csd_blend = 0.0
        # Ensure attributes exist even when using older implementations.
        if not hasattr(spectral_buffer, "use_csd"):
            spectral_buffer.use_csd = False
        if not hasattr(spectral_buffer, "csd_blend"):
            spectral_buffer.csd_blend = 0.0
        if sc.get("use_raw_replay", False):
            anchor_replay = ReplayBuffer(capacity=sc.get("raw_replay_capacity", 100))
            anchor_replay_weight = sc.get("raw_replay_weight", 1.0)
        spectral_reg = SpectralRegularizer(
            n_probes=sc.get("sfr_n_probes", 32),
            reg_weight=sc.get("sfr_weight", 1.0),
        )
        sfr_interval = sc.get("sfr_interval", 3)

        # Legacy components: NSP + FPSP (conditionally enabled)
        use_nsp = sc.get("use_nsp", True)
        use_fpsp = sc.get("use_fpsp", True)
        if use_fpsp:
            state_partitioner = StatePartitioner(
                protection_ratio=sc.get("protection_ratio", 0.5),
                criterion=sc.get("partition_criterion", "decay_rate"),
            )
        if use_nsp:
            null_projector = NullSpaceProjector(
                covariance_ema=sc.get("covariance_ema", 0.95),
                threshold=sc.get("null_space_threshold", 0.99),
                min_freedom_ratio=sc.get("min_freedom_ratio", 0.2),
                covariance_task_decay=sc.get("covariance_task_decay", 0.75),
            )

        # Novel components: FSHL + SGBD
        if sc.get("use_hint_loss", False):
            hint_loss_obj = FrequencyHintLoss(
                weight=sc.get("hint_weight", 1.0),
                decay=sc.get("hint_decay", 0.7),
            )
            hint_interval = sc.get("hint_interval", 3)
            print(f"    FSHL: weight={hint_loss_obj.weight}, decay={hint_loss_obj.decay}, interval={hint_interval}")
        if sc.get("use_sgbd", False):
            sgbd_obj = SpectralGradientDecomposer(
                protection_strength=sc.get("sgbd_protection_strength", 0.7),
                min_scale=sc.get("sgbd_min_scale", 0.1),
            )
            print(f"    SGBD: strength={sgbd_obj.protection_strength}, min_scale={sgbd_obj.min_scale}")

        spectracl_ewc = EWC(
            lambda_ewc=sc.get("ewc_lambda", 100.0),
        ) if sc.get("use_ewc", True) else None

        # Adaptive protection: store base values for scaling
        adaptive_protection = sc.get("adaptive_protection", False)
        adaptive_sensitivity = sc.get("adaptive_sensitivity", 2.0)
        base_protection_ratio = sc.get("protection_ratio", 0.5)
        base_min_freedom = sc.get("min_freedom_ratio", 0.2)
        base_ewc_lambda = sc.get("ewc_lambda", 100.0)

        print(f"    SIAR temperature: {spectral_buffer.siar_temperature}")
        print(f"    FDRL weight: {spectral_buffer.fdrl_weight}")
        print(f"    SIAR MSE weight: {getattr(spectral_buffer, 'siar_mse_weight', 0.0)}")
        if anchor_replay is not None:
            print(
                f"    Anchor replay: enabled, capacity={anchor_replay.capacity}, "
                f"weight={anchor_replay_weight}"
            )
        else:
            print("    Anchor replay: disabled")
        print(f"    SFR weight: {spectral_reg.reg_weight}, probes: {spectral_reg.n_probes}, interval: {sfr_interval}")
        print(f"    Buffer capacity: {spectral_buffer.capacity}")
        print(f"    Replay steps/task: 10 * n_tasks_stored")
        if use_csd and getattr(spectral_buffer, "use_csd", False):
            print(f"    CSD: enabled, blend={getattr(spectral_buffer, 'csd_blend', 0.0)}")
        elif use_csd:
            print("    CSD: requested but unavailable in this SpectralReplayBuffer version")
        if null_projector is not None:
            print(f"    NSP covariance_task_decay: {null_projector.covariance_task_decay}")
        else:
            print("    NSP: disabled")
        if state_partitioner is None:
            print("    FPSP: disabled")

    if method == "adaptivecl":
        ac = cfg.get("adaptivecl", {})
        adaptive_cl = AdaptiveCLBuffer(
            capacity=ac.get("capacity", 500),
            soft_capacity=ac.get("soft_capacity", 64),
            memory_batch_size=ac.get("memory_batch_size", tc.get("batch_size", 32)),
            soft_batch_size=ac.get("soft_batch_size", tc.get("batch_size", 32) // 2),
            alpha=ac.get("alpha", 0.5),
            beta=ac.get("beta", 0.5),
            gamma=ac.get("gamma", 0.5),
        )
        print(
            "    AdaptiveCL: "
            f"capacity={adaptive_cl.capacity}, "
            f"soft_capacity={adaptive_cl.soft_capacity}, "
            f"alpha={adaptive_cl.alpha}, beta={adaptive_cl.beta}, gamma={adaptive_cl.gamma}"
        )

    if method in ("clser", "clser_ifr"):
        cc = cfg.get("clser", {})
        clser_obj = CLSERAdapter(
            model=model,
            buffer_capacity=cc.get("capacity", 500),
            reg_weight=cc.get("reg_weight", 0.1),
            plastic_model_alpha=cc.get("plastic_model_alpha", 0.99),
            stable_model_alpha=cc.get("stable_model_alpha", 0.999),
            plastic_model_update_freq=cc.get("plastic_model_update_freq", 0.9),
            stable_model_update_freq=cc.get("stable_model_update_freq", 0.7),
        )
        print(
            "    CLS-ER: "
            f"capacity={clser_obj.buffer.capacity}, "
            f"reg_weight={clser_obj.reg_weight}, "
            f"alpha_p={clser_obj.plastic_model_alpha}, "
            f"alpha_s={clser_obj.stable_model_alpha}"
        )

    results_matrix = []
    profile_tasks = []
    # Case-based snapshot routing: evaluated alongside the base method on the same snapshots.
    router = None
    if case_sets:
        router = SnapshotRouter([tl.dataset for tl in test_loaders], cfg["data"].get("target_indices"),
                                device, tc["batch_size"], k=cfg.get("routing", {}).get("k", 10))
        routing_matrices = {}

    for task_id in range(n_tasks):
        print(f"\n  --- Task {task_id} ({len(train_loaders[task_id])} batches) ---")

        # Get replay loader from previous tasks (for Replay baseline)
        replay_loader = replay.get_loader(tc["batch_size"]) if replay is not None else None

        # SpectraCL: update protected param names from state partitioner
        if null_projector is not None and state_partitioner is not None and state_partitioner.is_active:
            protected_names = state_partitioner.get_protected_param_names(model)
            null_projector.protected_param_names = protected_names
            print(f"    SpectraCL: protecting {len(protected_names)} parameter groups")

        if method == "spectracl":
            components = []
            if spectral_buffer and spectral_buffer.size > 0:
                components.append(f"SIAR+FDRL({spectral_buffer.size} samples)")
            if spectral_reg and spectral_reg.is_active:
                components.append("SFR")
            if null_projector and null_projector.is_active:
                components.append("NSP")
            if state_partitioner and state_partitioner.is_active:
                components.append("FPSP")
            if spectracl_ewc and spectracl_ewc.is_registered:
                components.append("EWC")
            if hint_loss_obj and hint_loss_obj.is_active:
                components.append("FSHL")
            if sgbd_obj and sgbd_obj.is_active:
                components.append("SGBD")
            print(f"    Active: {', '.join(components) if components else 'none (first task)'}")

        if method == "adaptivecl":
            active = ["dual-buffer"]
            if adaptive_cl is not None and adaptive_cl.has_previous_model:
                active.append("hint-regularisation")
            print(f"    Active: {', '.join(active)}")

        if clser_obj is not None:
            print(
                "    Active: replay + dual-EMA consistency "
                f"(buffer={clser_obj.buffer.size})"
            )

        # SpectraCL: per-task LR decay to reduce forgetting pressure
        if method == "spectracl" and task_id > 0:
            lr_decay = sc.get("lr_decay_per_task", 0.9)
            new_lr = tc["learning_rate"] * (lr_decay ** task_id)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"    LR: {new_lr:.6f} (decay={lr_decay})")

        # SpectraCL: adaptive protection based on regime divergence
        if method == "spectracl" and adaptive_protection and task_id > 0 and spectral_buffer.n_tasks_stored > 0:
            divergence = spectral_buffer.compute_data_divergence(
                train_loaders[task_id], device
            )
            adapt_factor = 1.0 / (1.0 + adaptive_sensitivity * divergence)

            # Scale protection ratio down when divergence is high
            if state_partitioner is not None:
                state_partitioner.protection_ratio = base_protection_ratio * adapt_factor
            # Increase freedom when divergence is high
            if null_projector is not None:
                null_projector.min_freedom_ratio = min(
                    0.8, base_min_freedom + (1.0 - base_min_freedom) * (1.0 - adapt_factor)
                )
            # Scale EWC lambda down when divergence is high
            if spectracl_ewc is not None:
                spectracl_ewc.lambda_ewc = base_ewc_lambda * adapt_factor

            # Recompute masks and projectors with adapted values
            if state_partitioner is not None and state_partitioner.is_active:
                state_partitioner.update_partition(model)
                if null_projector is not None:
                    protected_names = state_partitioner.get_protected_param_names(model)
                    null_projector.protected_param_names = protected_names
            if null_projector is not None and null_projector.is_active:
                null_projector.recompute_projectors()

            print(f"    Adaptive protection: divergence={divergence:.4f}, factor={adapt_factor:.3f}")
            if state_partitioner is not None:
                print(f"      FPSP ratio: {base_protection_ratio:.2f} -> {state_partitioner.protection_ratio:.2f}")
            if null_projector is not None:
                print(f"      NSP freedom: {base_min_freedom:.2f} -> {null_projector.min_freedom_ratio:.2f}")
            if spectracl_ewc is not None:
                print(f"      EWC lambda: {base_ewc_lambda:.1f} -> {spectracl_ewc.lambda_ewc:.1f}")

        if ifr_obj is not None:
            ifr_obj.begin_task(train_loaders[task_id], device)
        if spectral_weighter is not None:
            spectral_weighter.begin_task(train_loaders[task_id], device)

        # Profiling: time and peak memory of this regime's training, measured with explicit
        # synchronisation so asynchronous GPU work is attributed to the right phase.
        cuda = str(device).startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_train0 = time.perf_counter()

        # Train
        # Compute replay steps: scale with number of stored tasks
        current_replay_steps = 0
        if method == "spectracl" and spectral_buffer is not None and spectral_buffer.n_tasks_stored > 0:
            current_replay_steps = 10 * spectral_buffer.n_tasks_stored
            print(f"    Replay steps/epoch: {current_replay_steps}")

        train_task(
            model=model,
            train_loader=train_loaders[task_id],
            optimizer=optimizer,
            device=device,
            epochs=epochs_per_task,
            grad_clip=tc["grad_clip"],
            ewc=ewc if ewc is not None else spectracl_ewc,
            si=si,
            mas=mas,
            replay_loader=replay_loader,
            derpp=derpp,
            fsnet_adapter=fsnet_adapter,
            spectral_buffer=spectral_buffer,
            anchor_replay=anchor_replay,
            anchor_replay_weight=anchor_replay_weight,
            null_projector=null_projector,
            state_partitioner=state_partitioner,
            spectral_reg=spectral_reg,
            sfr_interval=sfr_interval,
            replay_steps=current_replay_steps,
            hint_loss=hint_loss_obj,
            hint_interval=hint_interval,
            sgbd=sgbd_obj,
            adaptive_cl=adaptive_cl,
            clser=clser_obj,
            ifr=ifr_obj,
            spectral_weighter=spectral_weighter,
            mir=mir_obj,
            agem=agem_obj,
            verbose=True,
        )
        if cuda:
            torch.cuda.synchronize()
        t_train1 = time.perf_counter()

        if spectral_weighter is not None:
            spectral_weighter.end_task(train_loaders[task_id])

        if method.startswith("clfd") and task_id == 0:
            model.freeze_encoder()  # CLFD freezes its frequency encoder after the first task

        if ifr_obj is not None:
            ifr_obj.end_task(train_loaders[task_id])
            print(f"    IFR buffer: {ifr_obj.size} samples, {ifr_obj.memory_bytes() / 1e6:.2f} MB")

        # Post-task updates
        if fsnet_adapter is not None:
            fsnet_adapter.end_regime(model)

        if ewc is not None:
            print("    Computing Fisher information...")
            ewc.register_regime(model, train_loaders[task_id], device)

        if mas is not None:
            print("    Computing MAS importance...")
            mas.register_regime(model, train_loaders[task_id], device)

        if replay is not None:
            replay.add_from_loader(train_loaders[task_id])
            print(f"    Replay buffer size: {replay.size}")

        if mir_obj is not None:
            mir_obj.buffer.add_from_loader(train_loaders[task_id])
        if agem_obj is not None:
            agem_obj.buffer.add_from_loader(train_loaders[task_id])

        if derpp is not None:
            derpp.add_from_loader(model, train_loaders[task_id], device)
            print(f"    DER++ buffer size: {derpp.size}")

        if method == "adaptivecl" and adaptive_cl is not None:
            adaptive_cl.snapshot_previous_model(model)
            print(
                f"    AdaptiveCL main memory: {adaptive_cl.memory_size}, "
                f"soft memory: {adaptive_cl.soft_size}"
            )

        if clser_obj is not None:
            clser_obj.buffer.add_from_loader(train_loaders[task_id])
            print(f"    CLS-ER replay buffer size: {clser_obj.buffer.size}")

        # SpectraCL post-task updates
        if method == "spectracl":
            # 1. SMC: curate replay buffer with spectral diversity
            spectral_buffer.add_from_loader_smc(train_loaders[task_id], device)
            print(f"    SMC: {spectral_buffer.size} samples across {spectral_buffer.n_tasks_stored} tasks")

            # 1b. Optional raw anchor replay buffer
            if anchor_replay is not None:
                anchor_replay.add_from_loader(train_loaders[task_id])
                print(f"    Anchor replay: {anchor_replay.size} samples")

            # 2. Store task spectral profile for FDRL weighting
            spectral_buffer.store_task_profile(train_loaders[task_id], device)

            # 3. SFR: snapshot model's frequency response
            sfr_profile = spectral_buffer.task_profiles[-1] if spectral_buffer.task_profiles else None
            spectral_reg.snapshot(
                model, dc["seq_len"], num_features, device,
                spectral_profile=sfr_profile,
            )
            print(f"    SFR: {len(spectral_reg._frozen)} task snapshot(s), {spectral_reg.n_probes} probes")

            # 4. Update state partition (FPSP) -- if enabled
            if state_partitioner is not None:
                state_partitioner.update_partition(model, train_loaders[task_id], device)
                pstats = state_partitioner.get_partition_stats()
                for lname, lstats in pstats.items():
                    short = lname.split(".")[-2] + "." + lname.split(".")[-1]
                    print(f"    FPSP {short}: {lstats['protected_dims']}/{lstats['total_dims']} protected")

            # 5. Update null-space projector (NSP) -- if enabled
            if null_projector is not None:
                if state_partitioner is not None:
                    protected_names = state_partitioner.get_protected_param_names(model)
                    null_projector.protected_param_names = protected_names
                null_projector.update_subspace(model, train_loaders[task_id], device)
                nstats = null_projector.get_projection_stats()
                for lname, lstats in nstats.items():
                    if lstats["total_dim"] > 0:
                        short = lname.split(".")[-2] + "." + lname.split(".")[-1] if "." in lname else lname
                        print(f"    NSP {short}: {lstats['null_dim']}/{lstats['total_dim']} free dims ({lstats['freedom_ratio']:.1%})")

            # 6. FSHL: freeze model copy for hint distillation
            if hint_loss_obj is not None:
                hint_loss_obj.snapshot(model)
                hint_loss_obj.update_frequency_weights(spectral_buffer.task_profiles)
                print(f"    FSHL: {len(hint_loss_obj._frozen_models)} frozen model(s)")

            # 7. SGBD: update spectral importance for gradient scaling
            if sgbd_obj is not None:
                sgbd_obj.update_importance(spectral_buffer.task_profiles)
                print(f"    SGBD: importance spectrum updated ({sgbd_obj._n_tasks} tasks)")

            # 8. Update EWC (supplementary)
            if spectracl_ewc is not None:
                spectracl_ewc.register_regime(model, train_loaders[task_id], device)

        if cuda:
            torch.cuda.synchronize()
        profile_tasks.append({
            "task": task_id,
            "train_s": t_train1 - t_train0,
            "boundary_s": time.perf_counter() - t_train1,   # end-of-task updates (Fisher, buffers, snapshots)
            "steps": len(train_loaders[task_id]) * epochs_per_task,
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan"),
        })

        # Evaluate on ALL tasks
        eval_results = evaluate_all_tasks(model, test_loaders, device)
        results_matrix.append(eval_results)

        for r_id, res in enumerate(eval_results):
            status = "<-current" if r_id == task_id else ("<-old" if r_id < task_id else "<-future")
            print(f"    Task {r_id} test MSE: {res['mse']:.6f}  MAE: {res['mae']:.6f}  {status}")

        if router is not None:
            router.add(model, case_sets[task_id], train_loaders[task_id].dataset)
            rows = router.evaluate()
            for v, r in rows.items():
                routing_matrices.setdefault(v, []).append(r)
            gap = max(abs(a["mse"] - b["mse"]) for a, b in zip(rows["final"], eval_results))
            d = router.diag[-1]
            seen = slice(0, task_id + 1)
            print(f"    Routing: selected={d['selected']}  regime acc knn={d['regime_acc_knn']:.2f} "
                  f"dlda={d['regime_acc_dlda']:.2f} buf={d['regime_acc_buf']:.2f}  |final-check|={gap:.2e}")
            print("    Routing seen-avg MSE: " + "  ".join(
                f"{v}={np.mean([x['mse'] for x in r[seen]]):.4f}" for v, r in rows.items()))

    # Efficiency profile: stored state besides the model, model size, inference latency.
    method_state = [ewc, si, mas, replay, derpp, fsnet_adapter, spectral_buffer, anchor_replay,
                    null_projector, state_partitioner, spectracl_ewc, spectral_reg, hint_loss_obj,
                    sgbd_obj, adaptive_cl, clser_obj, ifr_obj, spectral_weighter, mir_obj, agem_obj]
    if router is not None:
        # Deployable knowledge base (rule selected on validation: sstack_global): weight fit time,
        # boundary update time, memory, inference latency, and a check against the in-run value.
        skb, fit_s = router.deployable()
        check = evaluate_all_tasks(skb, test_loaders, device)
        gap = max(abs(a["mse"] - b["mse"]) / b["mse"]
                  for a, b in zip(check, routing_matrices["sstack_global"][-1]))
        skb_profile = {"update_s": router.update_seconds, "fit_s": fit_s,
                       "active_snapshots": len(skb.snapshots),
                       "params_unique": sum(p.numel() for p in {p.data_ptr(): p for p in skb.parameters()}.values()),
                       "infer_ms_per_batch": inference_latency_ms(skb, test_loaders[0], device),
                       "check_rel_gap": gap}
        print(f"    SKB deployable: {len(skb.snapshots)} active snapshots, fit {fit_s:.3f}s, "
              f"latency {skb_profile['infer_ms_per_batch']:.2f} ms/batch, |rel. gap| {gap:.1e}")
        del skb
    profile = {
        "routing_bytes": router.memory_bytes() if router is not None else None,
        "skb": skb_profile if router is not None else None,
        "tasks": profile_tasks,
        "stored_bytes": stored_state_bytes(method_state, exclude=model),
        "params": count_parameters(model),
        "params_total": sum(p.numel() for p in model.parameters()),   # incl. frozen (e.g. CLFD encoder)
        "infer_ms_per_batch": inference_latency_ms(model, test_loaders[0], device),
        "infer_batch_size": test_loaders[0].batch_size,
    }

    # Collect predictions for plotting
    predictions = collect_predictions(model, test_loaders, device)

    out = {
        "method": method,
        "results_matrix": results_matrix,
        "final_results": results_matrix[-1],
        "predictions": predictions,
        "profile": profile,
    }
    if router is not None:
        out["routing"] = routing_matrices
        out["routing_diag"] = router.diag
        if dump_path:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            router.dump(dump_path)
    return out

# ===================================================================
# Compute continual learning metrics
# ===================================================================
def compute_cl_metrics(results_matrix: list[list[dict]], n_tasks: int) -> dict:
    final = results_matrix[-1]
    avg_mse = np.mean([r["mse"] for r in final])
    avg_mae = np.mean([r["mae"] for r in final])

    forgetting_values = []
    for r in range(n_tasks - 1):
        best_mse = min(results_matrix[t][r]["mse"] for t in range(r, len(results_matrix)))
        final_mse = results_matrix[-1][r]["mse"]
        forgetting_values.append(final_mse - best_mse)

    avg_forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    bwt_values = []
    for r in range(n_tasks - 1):
        after_r = results_matrix[r][r]["mse"]
        final_r = results_matrix[-1][r]["mse"]
        bwt_values.append(after_r - final_r)

    bwt = np.mean(bwt_values) if bwt_values else 0.0

    return {"avg_mse": avg_mse, "avg_mae": avg_mae, "avg_forgetting": avg_forgetting, "bwt": bwt}


# ===================================================================
# Print results table
# ===================================================================
def print_results_table(all_results: list[dict], n_tasks: int):
    print("\n" + "=" * 80)
    print("CONTINUAL LEARNING RESULTS SUMMARY")
    print("=" * 80)

    header = f"{'Method':<12} {'Avg MSE':<10} {'Avg MAE':<10} {'Forgetting':<12} {'BWT':<10}"
    regime_headers = "  ".join([f"T{i} MSE" for i in range(n_tasks)])
    print(f"{header}  {regime_headers}")
    print("-" * 80)

    for result in all_results:
        method = result["method"]
        final = result["final_results"]

        if method == "joint":
            avg_mse = np.mean([r["mse"] for r in final])
            avg_mae = np.mean([r["mae"] for r in final])
            regime_vals = "  ".join([f"{r['mse']:.4f}" for r in final])
            print(f"{method:<12} {avg_mse:<10.6f} {avg_mae:<10.6f} {'N/A':<12} {'N/A':<10}  {regime_vals}")
        else:
            metrics = compute_cl_metrics(result["results_matrix"], n_tasks)
            regime_vals = "  ".join([f"{r['mse']:.4f}" for r in final])
            print(f"{method:<12} {metrics['avg_mse']:<10.6f} {metrics['avg_mae']:<10.6f} "
                  f"{metrics['avg_forgetting']:<12.6f} {metrics['bwt']:<10.6f}  {regime_vals}")

    print("=" * 80)
    print("Forgetting > 0 means performance degraded. BWT > 0 means old tasks improved.")
    print("=" * 80)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="SpectraCL Continual Learning Experiments")
    parser.add_argument("--config", type=str, default="configs/continual_etth1_h24.yaml")
    parser.add_argument("--data", type=str, default="data/ETTh1.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["finetune", "ewc", "replay", "spectracl"],
                        choices=["finetune", "ewc", "si", "mas", "replay", "derpp",
                                 "fsnet", "spectracl", "adaptivecl", "clser", "joint",
                                 "ifr", "ifr_uniform", "derpp_ifr", "clser_ifr", "mir", "agem",
                                 "fredf", "clfd", "clfd_ifr", *GATED_METHODS,
                                 *IFR_ABLATIONS])
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Run every method once per seed (default: config seed)")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--seq_len", type=int, default=None, help="Override data.seq_len")
    parser.add_argument("--pred_len", type=int, default=None, help="Override data.pred_len")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs per task")
    parser.add_argument("--holdout", choices=["test", "val"], default="test",
                        help="'val' evaluates on a validation split carved from each task's "
                             "training segment (for tuning); 'test' is the final evaluation")
    parser.add_argument("--set", type=str, nargs="*", default=[],
                        help="Override config values, e.g. ifr.kappa=8 training.learning_rate=5e-4")
    args = parser.parse_args()

    resolved_config_path = os.path.abspath(args.config)
    cfg = load_config(resolved_config_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config path: {resolved_config_path}")

    for item in args.set:
        key, value = item.split("=", 1)
        node = cfg
        *path, leaf = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    if args.seq_len is not None:
        cfg["data"]["seq_len"] = args.seq_len
    if args.pred_len is not None:
        cfg["data"]["pred_len"] = args.pred_len
    if args.epochs is not None:
        cfg["training"]["epochs_per_regime"] = args.epochs

    dc = cfg["data"]
    tc = cfg["training"]
    n_tasks = args.n_tasks
    task_split = dc.get("regime_split", dc.get("task_split", "equal"))

    print("=" * 60)
    print(f"SpectraCL Continual Learning Experiments")
    print(f"Dataset: {args.data}")
    print(f"Tasks: {n_tasks} ({task_split} split)")
    print(f"Methods: {', '.join(args.methods)}")
    print("=" * 60)

    if "spectracl" in cfg:
        sc = cfg["spectracl"]
        print(
            "Loaded spectracl cfg: "
            f"capacity={sc.get('capacity', 500)}, "
            f"fdrl_weight={sc.get('fdrl_weight', 1.0)}, "
            f"siar_temperature={sc.get('siar_temperature', 0.5)}, "
            f"n_fft_fingerprint={sc.get('n_fft_fingerprint', 48)}"
        )

    set_seed(tc["seed"])
    case_ratio = cfg.get("routing", {}).get("case_ratio", 0.0)
    case_sets = [] if case_ratio > 0 else None
    feature_cols = dc.get("feature_cols", None)
    target_cols = dc.get("target_cols", None)

    if dc.get("domains"):
        # Cross-domain sequence: one univariate task per entry; --data is the data root.
        dc["target_indices"] = None
        train_loaders, test_loaders, scaler = build_multidomain_task_data(
            dc["domains"], dc["seq_len"], dc["pred_len"], tc["batch_size"],
            holdout=args.holdout, data_root=args.data,
            case_ratio=case_ratio, case_sets=case_sets,
        )
        print("Domains: " + " -> ".join(f"{d['csv']}:{d['column']}" for d in dc["domains"]))
    else:
        dc["target_indices"] = resolve_target_indices(
            args.data, feature_cols, target_cols, dc.get("date_col", "date"))
        print(f"Target indices for RevIN de-normalisation: {dc['target_indices']}")

        train_loaders, test_loaders, scaler = build_task_data(
            csv_path=args.data,
            seq_len=dc["seq_len"],
            pred_len=dc["pred_len"],
            n_tasks=n_tasks,
            batch_size=tc["batch_size"],
            feature_cols=feature_cols,
            target_cols=target_cols,
            date_col=dc.get("date_col", "date"),
            task_split=task_split,
            holdout=args.holdout,
            case_ratio=case_ratio, case_sets=case_sets,
        )
    print(f"Evaluation split: {args.holdout}")
    if case_sets:
        print(f"Case segments (held out from training): {[len(c) for c in case_sets]} windows")

    n_tasks = len(train_loaders)
    print(f"Effective tasks: {n_tasks}")

    joint_loader = None
    if "joint" in args.methods and not dc.get("domains"):
        joint_loader = build_joint_loader(
            csv_path=args.data, seq_len=dc["seq_len"], pred_len=dc["pred_len"],
            n_tasks=n_tasks, batch_size=tc["batch_size"],
            feature_cols=feature_cols, target_cols=target_cols,
            date_col=dc.get("date_col", "date"), task_split=task_split,
        )

    sample_x, sample_y = next(iter(train_loaders[0]))
    num_features = sample_x.shape[-1]
    output_features = sample_y.shape[-1]
    print(f"Input features: {num_features}, Output features: {output_features}")

    for i, (tl, te) in enumerate(zip(train_loaders, test_loaders)):
        print(f"  Task {i}: {len(tl)} train batches, {len(te)} test batches")

    # Run all methods for every seed -- re-seed before each
    seeds = args.seeds if args.seeds is not None else [tc["seed"]]
    os.makedirs(args.out_dir, exist_ok=True)
    # Unique per process: parallel jobs may share an output directory and start in the same second.
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{'-'.join(args.methods)}_pid{os.getpid()}"
    save_data = []
    pred_data = {}
    for seed in seeds:
        all_results = []
        for method in args.methods:
            set_seed(seed)
            t0 = time.time()
            result = run_method(
                method=method, cfg=cfg,
                train_loaders=train_loaders, test_loaders=test_loaders,
                joint_loader=joint_loader, device=device,
                num_features=num_features, output_features=output_features,
                case_sets=case_sets,
                dump_path=os.path.join(args.out_dir, "snapshots", f"{stamp}_{method}_seed{seed}.npz")
                if case_sets else None,
            )
            elapsed = time.time() - t0
            result["time"] = elapsed
            all_results.append(result)
            print(f"\n  {method} (seed {seed}) completed in {elapsed:.1f}s")

            # Runs with a held-out case base train on less data: keep them distinguishable.
            name = f"{method}@cases" if case_sets else method
            entry = {"method": name, "seed": seed, "time": elapsed, "holdout": args.holdout,
                     "config": resolved_config_path, "overrides": args.set,
                     "seq_len": dc["seq_len"], "pred_len": dc["pred_len"],
                     "final_results": result["final_results"]}
            if method != "joint":
                entry["metrics"] = compute_cl_metrics(result["results_matrix"], n_tasks)
            entry["results_matrix"] = result["results_matrix"]
            if "profile" in result:
                entry["profile"] = result["profile"]
            if "routing" in result:
                entry["case_ratio"] = case_ratio
                entry["routing"] = result["routing"]
                entry["routing_diag"] = result["routing_diag"]
            save_data.append(entry)
            if "predictions" in result:
                pred_data[f"{method}_seed{seed}"] = result["predictions"]

            # Save after every method so partial runs are never lost
            save_path = os.path.join(args.out_dir, f"continual_results_{stamp}.json")
            with open(save_path, "w") as f:
                json.dump(save_data, f, indent=2, default=str)

        print(f"\nSeed {seed}:")
        print_results_table(all_results, n_tasks)

    print(f"\nResults saved to: {save_path}")
    pred_path = os.path.join(args.out_dir, f"continual_predictions_{stamp}.json")
    with open(pred_path, "w") as f:
        json.dump(pred_data, f)
    print(f"Predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()