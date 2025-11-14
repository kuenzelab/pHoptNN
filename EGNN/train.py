#!/usr/bin/env python3
"""
Train an EGNN model to predict enzyme pH optima.

Highlights
---------
- Works with preprocessed, batched datasets produced by `create_pyg_dataset.py`.
- Supports single train/val split or K-fold cross-validation.
- Label-Distribution Smoothing (LDS, Yang et al.) and class-balanced weighting.
- Graceful early stopping and checkpointing.
- Minimal, dependency-friendly logging.

Expected inputs
---------------
`--root_dir`
    Root that contains `train/raw/train.csv` and a saved batched dataset file
    named `egnn_train_dataset_bs_<BATCH>.pt` (created by `create_pyg_dataset.py`).

`--df_individuals`
    CSV with one row of hyperparameters per model (grid). The row selected by
    `--idx_individual` is used. Expected columns (with reasonable defaults used
    if missing):
      - num_epochs, batch_size, lr, weight_decay
      - num_bins, loss_weighting, ks, sigma, weighting_factor
      - num_layers, num_node_features, attention

Outputs
-------
- Console/file logs.
- Optional: per-epoch loss CSV (`--losses_csv_path`).
- Optional: best model checkpoint(s) (`--save_models_path`).

"""
from __future__ import annotations

import argparse
import logging
import os
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import convolve1d
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Subset, random_split
from tqdm import tqdm

from create_pyg_dataset import EGNNBatchDataset
from qm9 import utils as qm9_utils
from qm9 import dataset
from models.egnn import EGNN
from utils_Yang_lds import get_lds_kernel_window, weighted_mse_loss

# --------------------------- Repro & defaults ---------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
CHARGE_POWER = 2
PROPERTY_COL = "ph_opti"  # internal normalized label key in EGNNBatchDataset dict
SPLIT_FRACTION = 0.875     # single split: fraction for training
VAL_EVERY = 1
NUM_WORKERS = max(0, min(6, (os.cpu_count() or 2) - 1))
EDGE_DIM = 5               # edge feature size produced by create_pyg_dataset

# ------------------------------ CLI ------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one EGNN model (single split or K-Fold)")
    p.add_argument("--root_dir", required=True,
                   help="Root directory containing egnn_train_dataset_bs_*.pt and train/raw/train.csv")
    p.add_argument("--df_individuals", required=True,
                   help="CSV with hyperparameter grid. One model per row.")
    p.add_argument("--idx_individual", type=int, required=True,
                   help="Row index to pick from the hyperparameter CSV.")

    p.add_argument("--early_stopping_patience", type=int, default=20,
                   help="Epochs to wait for val MSE improvement before stopping.")
    p.add_argument("--kfold", type=int, default=None,
                   help="If set, perform K-fold cross-validation with this many folds.")
    p.add_argument("--losses_csv_path", type=str, default=None,
                   help="Directory to write per-epoch losses or CV summary.")
    p.add_argument("--save_models_path", type=str, default=None,
                   help="Directory to save best checkpoint(s).")
    p.add_argument("--log_path", type=str, default=None,
                   help="Optional path to a log file. If omitted, logs are console-only.")
    return p

# --------------------------- Logging utils -----------------------------

def setup_logging(log_path: str | None) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler()]
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)

# --------------------------- Helpers -----------------------------------

def config_from_csv_row(csv_path: str, idx: int) -> Dict:
    df = pd.read_csv(csv_path)
    row = df.iloc[idx].to_dict()
    return row


def get_bin_idx(label_norm: float, *, num_bins: int, y_mean: float, y_std: float) -> int:
    """Return discretization bin for a standardized label value."""
    lower = (0 - y_mean) / y_std
    upper = (14 - y_mean) / y_std
    bin_size = (upper - lower) / num_bins
    return int(np.clip((label_norm - lower) / bin_size, 0, num_bins - 1))

# ----------------------- Train / Val steps -----------------------------

def forward_one_batch(batch: Dict, *, model: nn.Module, charge_scale: torch.Tensor,
                      y_mean: torch.Tensor, y_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward a single pre-batched item from EGNNBatchDataset.

    The dataset returns a dict of tensors for a full micro-batch.
    """
    batch_size, n_nodes, _ = batch["positions"].size()

    positions = batch["positions"].view(batch_size * n_nodes, -1).to(DEVICE, DTYPE)
    node_mask = batch["atom_mask"].view(batch_size * n_nodes, -1).to(DEVICE, DTYPE)
    one_hot   = batch["one_hot"].to(DEVICE, DTYPE)
    charges   = batch["charges"].to(DEVICE, DTYPE)

    # preprocessing must mirror training pipeline in qm9/utils
    nodes = qm9_utils.preprocess_input(one_hot, charges, CHARGE_POWER, charge_scale, DEVICE)
    nodes = nodes.view(batch_size * n_nodes, -1)

    edges     = batch["edges"].to(DEVICE)
    edge_attr = batch["edge_attr"].to(DEVICE, DTYPE)
    edge_mask = batch["edge_mask"].to(DEVICE, DTYPE)

    y_norm = batch[PROPERTY_COL].to(DEVICE, DTYPE)
    if y_norm.dim() == 0:
        y_norm = y_norm.unsqueeze(0)

    pred_norm = model(
        h0=nodes,
        x=positions,
        edges=edges,
        edge_attr=edge_attr,
        node_mask=node_mask,
        edge_mask=edge_mask,
        n_nodes=n_nodes,
    )
    return pred_norm.view(-1), y_norm.view(-1)


def train_one_epoch(epoch: int, loader, *, model: nn.Module, optimizer: optim.Optimizer,
                    scaler: GradScaler, charge_scale: torch.Tensor,
                    y_mean: torch.Tensor, y_std: torch.Tensor,
                    num_bins: int, bin_weight: torch.Tensor,
                    fold_num: int | None = None) -> float:
    model.train()
    running = 0.0
    tag = f"Fold {fold_num+1} " if fold_num is not None else ""
    for batch in tqdm(loader, desc=f"{tag}Epoch {epoch} [train]", leave=False):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            pred, y = forward_one_batch(batch, model=model, charge_scale=charge_scale,
                                         y_mean=y_mean, y_std=y_std)
            # compute LDS weights per-sample
            bin_idx = torch.tensor([
                get_bin_idx(v.item(), num_bins=num_bins, y_mean=y_mean.item(), y_std=y_std.item())
                for v in y
            ], device=DEVICE, dtype=torch.long)
            lw = bin_weight[bin_idx]
            loss = weighted_mse_loss(pred, y, lw)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item())
    return running / max(1, len(loader))


def validate(epoch: int, loader, *, model: nn.Module, charge_scale: torch.Tensor,
             y_mean: torch.Tensor, y_std: torch.Tensor,
             num_bins: int, bin_weight: torch.Tensor,
             fold_num: int | None = None) -> Tuple[float, float]:
    model.eval()
    loss_lds = 0.0
    loss_mse = 0.0
    tag = f"Fold {fold_num+1} " if fold_num is not None else ""
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{tag}Epoch {epoch} [val]", leave=False):
            pred_n, y_n = forward_one_batch(batch, model=model, charge_scale=charge_scale,
                                            y_mean=y_mean, y_std=y_std)
            pred_raw = pred_n * y_std + y_mean
            y_raw    = y_n * y_std + y_mean
            loss_mse += float(nn.functional.mse_loss(pred_raw, y_raw))

            bin_idx = torch.tensor([
                get_bin_idx(v.item(), num_bins=num_bins, y_mean=y_mean.item(), y_std=y_std.item())
                for v in y_n
            ], device=DEVICE, dtype=torch.long)
            lw = bin_weight[bin_idx]
            loss_lds += float(weighted_mse_loss(pred_n, y_n, lw))
    n = max(1, len(loader))
    return loss_lds / n, loss_mse / n

# ------------------------------- Main ----------------------------------

def main() -> None:
    args = build_argparser().parse_args()
    setup_logging(args.log_path)
    logging.info("Starting training run")

    # Load hyperparameters row
    cfg = config_from_csv_row(args.df_individuals, args.idx_individual)

    # Label stats from training CSV (used for LDS binning & denorm for metrics)
    train_csv = os.path.join(args.root_dir, "raw", "train.csv")
    df_meta = pd.read_csv(train_csv)
    y_vals = df_meta["ph_optimum"].to_numpy(dtype=float)
    y_mean_val, y_std_val = float(np.mean(y_vals)), float(np.std(y_vals))
    y_mean = torch.tensor(y_mean_val, dtype=DTYPE, device=DEVICE)
    y_std  = torch.tensor(y_std_val,  dtype=DTYPE, device=DEVICE)

    # Charge scale from qm9 dataset utilities (kept for parity with training code)
    _, charge_scale_val = dataset.retrieve_dataloaders(batch_size=int(cfg.get("batch_size", 8)),
                                                       num_workers=NUM_WORKERS)
    charge_scale = torch.tensor(float(charge_scale_val), dtype=DTYPE, device=DEVICE)

    # Load batched dataset
    pt_path = os.path.join(args.root_dir, f"egnn_train_dataset_bs_{int(cfg.get('batch_size', 8))}.pt")
    logging.info(f"Loading dataset: {pt_path}")
    big_ds = EGNNBatchDataset.load(pt_path)

    # Infer node feature dimension from a sample
    sample_nodes = qm9_utils.preprocess_input(big_ds[0]["one_hot"].to(DEVICE),
                                              big_ds[0]["charges"].to(DEVICE),
                                              CHARGE_POWER, charge_scale, DEVICE)
    in_node_nf = int(sample_nodes.size(-1))
    logging.info(f"Detected node feature size: {in_node_nf}")

    # Prepare LDS weights (or uniform)
    if str(cfg.get("loss_weighting", "none")).lower() == "lds_yang":
        num_bins = int(cfg.get("num_bins", 50))
        labels_norm = ((y_vals - y_mean_val) / (y_std_val + 1e-12)).tolist()
        bin_ids = [get_bin_idx(v, num_bins=num_bins, y_mean=y_mean_val, y_std=y_std_val) for v in labels_norm]
        counts = Counter(bin_ids)
        emp_dist = np.array([counts.get(i, 0) for i in range(num_bins)], dtype=float)
        ks = int(cfg.get("ks", 5))
        sigma = float(cfg.get("sigma", 2.0))
        kernel = get_lds_kernel_window("gaussian", ks=ks, sigma=sigma)
        eff_dist = convolve1d(emp_dist, weights=np.asarray(kernel), mode="constant")
        wf = float(cfg.get("weighting_factor", 1.0))
        bin_weight = torch.tensor([1.0 / (x ** wf + 1e-8) for x in eff_dist], dtype=DTYPE, device=DEVICE)
    else:
        num_bins = int(cfg.get("num_bins", 50))
        bin_weight = torch.ones(num_bins, dtype=DTYPE, device=DEVICE)

    # Helper to build a fresh model
    def build_model(use_attention: bool) -> nn.Module:
        return EGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=EDGE_DIM,
            hidden_nf=int(cfg.get("num_node_features", 128)),
            device=DEVICE,
            n_layers=int(cfg.get("num_layers", 4)),
            node_attr=0,
            attention=bool(use_attention),
        ).to(DEVICE)

    # ------------------------ K-Fold path ------------------------
    if args.kfold is not None and int(args.kfold) > 1:
        k = int(args.kfold)
        kfold = KFold(n_splits=k, shuffle=True, random_state=SEED)
        fold_summaries = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(big_ds)):
            logging.info("%s", "=" * 32 + f" FOLD {fold+1}/{k} " + "=" * 32)
            train_ds = Subset(big_ds, train_ids)
            val_ds   = Subset(big_ds, val_ids)

            model = build_model(bool(cfg.get("attention", False)))
            optimizer = optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)),
                                    weight_decay=float(cfg.get("weight_decay", 0.0)))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
            scaler = GradScaler()

            best_val_mse = float("inf")
            patience = 0

            for epoch in range(int(cfg.get("num_epochs", 200))):
                tr_loss = train_one_epoch(epoch, train_ds, model=model, optimizer=optimizer, scaler=scaler,
                                          charge_scale=charge_scale, y_mean=y_mean, y_std=y_std,
                                          num_bins=num_bins, bin_weight=bin_weight, fold_num=fold)
                val_loss, val_mse = validate(epoch, val_ds, model=model, charge_scale=charge_scale,
                                             y_mean=y_mean, y_std=y_std, num_bins=num_bins,
                                             bin_weight=bin_weight, fold_num=fold)
                scheduler.step(val_loss)
                logging.info(f"Fold {fold+1} Ep {epoch:03d} | Train {tr_loss:.6f} | Val(LDS) {val_loss:.6f} | Val(MSE) {val_mse:.6f}")

                improved = val_mse < best_val_mse - 1e-9
                if improved:
                    best_val_mse = val_mse
                    patience = 0
                    if args.save_models_path:
                        os.makedirs(args.save_models_path, exist_ok=True)
                        ckpt = os.path.join(args.save_models_path, f"model_fold{fold+1}_idx{args.idx_individual}.pt")
                        torch.save(model.state_dict(), ckpt)
                        logging.info("Saved new best checkpoint: %s", ckpt)
                else:
                    patience += 1

                if patience >= int(args.early_stopping_patience):
                    logging.info("Early stopping at epoch %d", epoch)
                    break

            fold_summaries.append({"fold": fold + 1, "best_val_mse": best_val_mse})

        # Summary & optional CSV
        df_summary = pd.DataFrame(fold_summaries)
        mean_mse = float(df_summary["best_val_mse"].mean())
        std_mse  = float(df_summary["best_val_mse"].std(ddof=0))
        logging.info("K-FOLD SUMMARY: mean MSE = %.6f ± %.6f", mean_mse, std_mse)
        if args.losses_csv_path:
            os.makedirs(args.losses_csv_path, exist_ok=True)
            out_csv = os.path.join(args.losses_csv_path, "losses_overview_kfold.csv")
            row = {**cfg, "idx_individual": args.idx_individual, "kfold": k,
                   "mean_val_loss": round(mean_mse, 8), "std_val_loss": round(std_mse, 8)}
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            logging.info("Saved CV summary to %s", out_csv)
        return

    # ------------------------ Single-split path ------------------------
    n_train = int(len(big_ds) * SPLIT_FRACTION)
    n_val   = len(big_ds) - n_train
    train_ds, val_ds = random_split(big_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    model = build_model(bool(cfg.get("attention", False)))
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)),
                            weight_decay=float(cfg.get("weight_decay", 0.0)))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    scaler = GradScaler()

    best_val_mse = float("inf")
    patience = 0
    train_losses, val_losses, val_mse_list = [], [], []

    for epoch in range(int(cfg.get("num_epochs", 200))):
        tr_loss = train_one_epoch(epoch, train_ds, model=model, optimizer=optimizer, scaler=scaler,
                                  charge_scale=charge_scale, y_mean=y_mean, y_std=y_std,
                                  num_bins=num_bins, bin_weight=bin_weight)
        val_loss, val_mse = validate(epoch, val_ds, model=model, charge_scale=charge_scale,
                                     y_mean=y_mean, y_std=y_std, num_bins=num_bins,
                                     bin_weight=bin_weight)
        scheduler.step(val_loss)
        logging.info(f"Epoch {epoch:03d} | Train {tr_loss:.6f} | Val(LDS) {val_loss:.6f} | Val(MSE) {val_mse:.6f}")

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_mse_list.append(val_mse)

        if val_mse < best_val_mse - 1e-9:
            best_val_mse = val_mse
            patience = 0
            if args.save_models_path:
                os.makedirs(args.save_models_path, exist_ok=True)
                ckpt = os.path.join(args.save_models_path, f"model_idx{args.idx_individual}.pt")
                torch.save(model.state_dict(), ckpt)
                logging.info("Saved new best checkpoint: %s", ckpt)
        else:
            patience += 1

        if patience >= int(args.early_stopping_patience):
            logging.info("Early stopping at epoch %d", epoch)
            break

    logging.info("Training finished. Best Val MSE: %.8f", best_val_mse)

    if args.losses_csv_path:
        os.makedirs(args.losses_csv_path, exist_ok=True)
        out_csv = os.path.join(args.losses_csv_path, f"losses_ind_{args.idx_individual}.csv")
        pd.DataFrame({
            "epoch": list(range(len(train_losses))),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_loss_MSE": val_mse_list,
        }).to_csv(out_csv, index=False)
        logging.info("Saved per-epoch losses to %s", out_csv)


if __name__ == "__main__":
    main()
