#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
pHoptNN — Enzyme pH Optimum Prediction (EGNN-based Inference)
===============================================================================

Overview
--------
This script predicts the optimal pH (pHopt) of enzyme structures using a trained
Equivariant Graph Neural Network (EGNN) checkpoint. It is designed for robust,
automated inference with minimal dependencies on internal utilities.

Inputs
------
--input_path        Path to a .pdb file or folder of .pdb files.
--model_weights     Path to trained EGNN weights (.pt).
--params_csv        Hyperparameter CSV from training.
--idx_individual    Row index in hyperparameter CSV.
--train_csv_path    CSV used to compute y_mean/y_std if not provided.
--y_mean, --y_std   Precomputed training dataset mean and std of pHopt.
--pqr_dir           Directory to read/write intermediate PQR files.
--save_attention_path  Directory to store exported attention CSVs.

Outputs
-------
- Prints per-file predicted pH optimum.
- Saves optional attention maps (`*_attention.csv`) under save_attention_path.
- PQRs are normalized automatically to ensure compatibility with legacy parsers.


Example
-------
    python predict_new.py \
        --input_path ./examples/1A0A.pdb \
        --model_weights ./weight/W_6_attn.pt \
        --params_csv ./EGNN/hyperparameters/Best_hp.csv \
        --idx_individual 6 \
        --train_csv_path ./pyg_datasets_connected/train/raw/train.csv \
        --y_mean 7.1956 --y_std 1.2302 \
        --pqr_dir ./pqr_files \
        --save_attention_path ./pred_out

Author: Raj
===============================================================================
"""
import os
import sys
import glob
import argparse
import warnings
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_sum, scatter_add
from tqdm import tqdm

# --- Silence warnings (incl. RDKit) early ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from rdkit import RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

# --- Project imports ---
from qm9.models import EGNN
from qm9 import utils as qm9_utils
from qm9 import dataset as qm9_dataset

from create_pyg_dataset import get_edge_features_from_rdkit
from utils import extract_pqr_data, build_edges_blockwise
from constants import ALL_ATOM_LABELS

# ----------------------------- Globals -----------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Captured attention per layer (edge gates after sigmoid)
_attention_cache: Dict[str, torch.Tensor] = {}

# ------------------------- Utility helpers -------------------------
def quiet_print(msg: str):
    print(msg, flush=True)

def to_bool(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "t")

def run_pdb2pqr(pdb_path: str, pqr_output_path: str):
    """Run pdb2pqr (AMBER) to generate PQR when needed."""
    os.makedirs(os.path.dirname(pqr_output_path) or ".", exist_ok=True)
    cmd = f"pdb2pqr30 --ff=AMBER {pdb_path} {pqr_output_path}"
    ret = os.system(cmd)
    if ret != 0 or not os.path.exists(pqr_output_path):
        raise FileNotFoundError(
            f"PDB2PQR failed for {os.path.basename(pdb_path)}. "
            f"Ensure pdb2pqr30 is installed and in PATH."
        )

def load_hparams(params_csv: str, idx: int) -> dict:
    """Read minimal hyperparameters needed to instantiate the EGNN."""
    df = pd.read_csv(params_csv, sep=None, engine="python")
    df.columns = [c.strip().strip('"').strip("'").lower() for c in df.columns]
    row = df.iloc[idx]
    return {
        "num_layers": int(row.get("num_layers", row.get("n_layers", 4))),
        "num_node_features": int(row.get("num_node_features", row.get("hidden_nf", 128))),
        "attention": to_bool(row.get("attention", row.get("use_attention", False))),
    }

def load_y_stats(train_csv_path: Optional[str], y_mean: Optional[float], y_std: Optional[float]) -> Tuple[float, float]:
    """Use explicit mean/std if given, else compute from training CSV (column 'ph_optimum')."""
    if y_mean is not None and y_std is not None:
        return float(y_mean), float(y_std)
    if not train_csv_path:
        raise ValueError("Provide --y_mean and --y_std, or a --train_csv_path to compute them.")
    df = pd.read_csv(train_csv_path)
    if "ph_optimum" not in df.columns:
        raise KeyError("Column 'ph_optimum' not found in train CSV.")
    y = df["ph_optimum"].astype(float).to_numpy()
    return float(y.mean()), float(y.std(ddof=0))

# ----------------------- Graph construction ------------------------
def process_structure_to_graph(pdb_path: str, pqr_path: str) -> Tuple[Data, List[dict]]:
    """
    Build a torch_geometric Data from PDB/PQR:
      - node features: [x,y,z, charge, onehot(atom)]
      - edges: from build_edges_blockwise(atom_entries)
      - edge_attr: RDKit (bond types + ring) if available else 5-dim distance bins
    Returns (graph, atom_entries_list)
    """
    # Edge features from RDKit (gracefully handle failures)
    edge_features_dict = {}
    used_rdkit = False
    try:
        edge_features_dict = get_edge_features_from_rdkit(pdb_path)
        used_rdkit = bool(edge_features_dict)
    except Exception:
        edge_features_dict = {}

    atom_entries = extract_pqr_data(pqr_path, less_aa=False)
    if not atom_entries:
        raise ValueError(f"Could not extract any atoms from {pqr_path}")

    positions, atom_labels, charges = [], [], []
    for entry in atom_entries:
        positions.append([entry["x"], entry["y"], entry["z"]])
        atom_labels.append(entry["atom_label"])
        charges.append(entry["charge"])

    one_hot = torch.tensor(
        [[1 if label == l else 0 for l in ALL_ATOM_LABELS] for label in atom_labels],
        dtype=DTYPE,
    )
    charges_tensor = torch.tensor(charges, dtype=DTYPE).unsqueeze(1)
    pos_tensor = torch.tensor(positions, dtype=DTYPE)

    node_features = torch.cat([pos_tensor, charges_tensor, one_hot], dim=1)
    edge_index = build_edges_blockwise(atom_entries)  # [2, E] long

    # Edge attributes (5-dim): RDKit if present; otherwise distance bins + ring=0
    def edge_feat_from_distance(u: int, v: int) -> List[float]:
        d = torch.norm(pos_tensor[u] - pos_tensor[v]).item()
        if d < 1.6:      bins = [1.0, 0.0, 0.0, 0.0]  # very short
        elif d < 3.0:    bins = [0.0, 1.0, 0.0, 0.0]  # short
        elif d < 5.0:    bins = [0.0, 0.0, 1.0, 0.0]  # medium
        else:            bins = [0.0, 0.0, 0.0, 1.0]  # long
        return bins + [0.0]  # ring flag unknown => 0

    edge_attr_list = []
    for i in range(edge_index.size(1)):
        u, v = edge_index[:, i].tolist()
        feat = edge_features_dict.get(tuple(sorted((u, v))))
        edge_attr_list.append(feat if feat is not None else edge_feat_from_distance(u, v))

    if used_rdkit:
        quiet_print("  • Edge features via RDKit (+ring flag).")
    else:
        quiet_print("  • Edge features via distance bins (fallback).")

    edge_attr = torch.tensor(edge_attr_list, dtype=DTYPE)
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return graph, atom_entries

def infer_in_dims_from_graph(graph: Data, charge_power: int, charge_scale: float) -> Tuple[int, int]:
    """Infer (in_node_nf, in_edge_nf) exactly as done in training code."""
    positions = graph.x[:, :3].to(DEVICE, DTYPE)
    charges = graph.x[:, 3].to(DEVICE, DTYPE)
    one_hot = graph.x[:, 4:].to(DEVICE, DTYPE)
    nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power,
                                       torch.tensor(charge_scale, device=DEVICE, dtype=DTYPE), DEVICE)
    in_node_nf = int(nodes.shape[-1])
    in_edge_nf = int(graph.edge_attr.size(1)) if graph.edge_attr is not None else 0
    return in_node_nf, in_edge_nf

# --------------------- Attention capture utilities ---------------------
def _att_hook_factory(layer_name: str):
    def hook(module, inp, out):
        # 'out' is the attention gate (after sigmoid) per edge for this layer
        _attention_cache[layer_name] = out.detach().cpu()
    return hook

def register_attention_hooks(model: nn.Module, num_layers: int):
    """Attach hooks to gcl_i.att_mlp if present (silently skip if absent)."""
    for i in range(num_layers):
        lname = f"gcl_{i}"
        layer = getattr(model, lname, None)
        if layer is None:
            continue
        if hasattr(layer, "att_mlp") and isinstance(layer.att_mlp, nn.Module):
            layer.att_mlp.register_forward_hook(_att_hook_factory(lname))

def compute_edge_attention_rollout() -> Optional[torch.Tensor]:
    """Average attention over layers → [E] (or None if no attention layers)."""
    if not _attention_cache:
        return None
    keys = sorted(_attention_cache.keys())
    edge_atts = [ _attention_cache[k].reshape(-1).float() for k in keys ]
    return torch.stack(edge_atts, dim=0).mean(0)  # [E]

def edge_to_node_attention(edge_att: torch.Tensor, edge_index: torch.Tensor,
                           n_nodes: int, agg: str = "mean") -> torch.Tensor:
    """
    Aggregate per-edge attention to per-node.
      agg='sum'  → sum of outgoing edge attentions
      agg='mean' → degree-normalized mean of outgoing edge attentions
    """
    row = edge_index[0]
    atom_sum = scatter_sum(edge_att, row, dim=0, dim_size=n_nodes)
    if agg == "sum":
        return atom_sum
    deg = scatter_add(torch.ones_like(edge_att), row, dim=0, dim_size=n_nodes)
    return atom_sum / (deg + 1e-8)

# --------------------------- Prediction core ---------------------------
def predict_one(graph: Data,
                atom_entries: List[dict],
                model: nn.Module,
                charge_power: int,
                charge_scale: float,
                y_mean: float,
                y_std: float,
                att_export: str,
                node_agg: str,
                save_dir: Optional[str],
                base_name: str) -> float:
    """
    Run a single forward pass, return raw prediction, and optionally export attention CSV(s).
    All outputs land in save_dir with filenames prefixed by base_name.
    """
    n_atoms = graph.x.size(0)

    positions = graph.x[:, :3].unsqueeze(0).to(DEVICE, DTYPE)       # [1, N, 3]
    charges   = graph.x[:, 3].unsqueeze(0).to(DEVICE, DTYPE)        # [1, N]
    one_hot   = graph.x[:, 4:].unsqueeze(0).to(DEVICE, DTYPE)       # [1, N, K]
    atom_mask = torch.ones(1, n_atoms, 1, dtype=DTYPE, device=DEVICE)

    nodes = qm9_utils.preprocess_input(one_hot.squeeze(0), charges.squeeze(0),
                                       charge_power,
                                       torch.tensor(charge_scale, device=DEVICE, dtype=DTYPE),
                                       DEVICE)

    edges     = graph.edge_index.to(DEVICE)
    edge_attr = graph.edge_attr.to(DEVICE, DTYPE)
    edge_mask = torch.ones(edges.size(1), 1, dtype=DTYPE, device=DEVICE)

    # clear attention cache for this forward
    _attention_cache.clear()

    with torch.no_grad():
        pred_norm = model(
            h0=nodes,
            x=positions.view(-1, 3),
            edges=edges,
            edge_attr=edge_attr,
            node_mask=atom_mask.view(-1, 1),
            edge_mask=edge_mask,
            n_nodes=n_atoms
        ).view(-1)

    pred_raw = pred_norm.item() * y_std + y_mean

    # --- Export attentions if requested ---
    if att_export in ("edge", "both", "node") and save_dir:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        edge_att_roll = compute_edge_attention_rollout()  # [E] or None
        if edge_att_roll is not None:

            if att_export in ("edge", "both"):
                # Save EDGE attention CSV (in save_dir)
                rows = []
                src = edges[0].cpu().numpy()
                dst = edges[1].cpu().numpy()
                ea = edge_att_roll.cpu().numpy()
                for e_idx in range(len(ea)):
                    u = int(src[e_idx])
                    v = int(dst[e_idx])
                    au = atom_entries[u]
                    av = atom_entries[v]
                    rows.append({
                        "edge_index": e_idx,
                        "src_atom": u,
                        "dst_atom": v,
                        "src_residue_name": au.get("residue_name", ""),
                        "src_residue_id":   au.get("residue_id", -1),
                        "src_atom_label":   au.get("atom_label", ""),
                        "dst_residue_name": av.get("residue_name", ""),
                        "dst_residue_id":   av.get("residue_id", -1),
                        "dst_atom_label":   av.get("atom_label", ""),
                        "attention_edge": float(ea[e_idx]),
                    })
                pd.DataFrame(rows).to_csv(
                    os.path.join(save_dir, f"{base_name}_edge_attention.csv"),
                    index=False
                )

            if att_export in ("node", "both"):
                # Aggregate to NODE attention (sum or mean)
                node_att = edge_to_node_attention(edge_att_roll.to(DEVICE), edges, n_atoms, agg=node_agg).cpu().numpy()
                # Normalize per-structure (baseline=1.0)
                na = node_att
                na_mean = float(np.mean(na)) if len(na) > 0 else 1.0
                na_min, na_max = float(np.min(na)), float(np.max(na))
                denom = (na_max - na_min) if (na_max - na_min) > 1e-8 else 1.0
                rows = []
                for i, entry in enumerate(atom_entries):
                    rows.append({
                        "atom_number": entry.get("atom_number", i),
                        "residue_name": entry.get("residue_name", ""),
                        "residue_id": entry.get("residue_id", -1),
                        "atom_label": entry.get("atom_label", ""),
                        "attention_score": float(na[i]),                          # raw per-node (sum/mean)
                        "attention_minmax01": float((na[i] - na_min) / denom),    # [0,1]
                        "attention_norm_baseline1": float(na[i] / (na_mean + 1e-8)), # baseline=1
                    })
                pd.DataFrame(rows).to_csv(
                    os.path.join(save_dir, f"{base_name}_attention.csv"),
                    index=False
                )

    return pred_raw

# ------------------------------ Main CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict pH optimum from PDB(s) with EGNN + attention export (edge/node)."
    )
    p.add_argument("--input_path", required=True,
                   help="A .pdb file or a directory with .pdb files.")
    p.add_argument("--model_weights", default="ATTbest_model_6/weight/Attention_model_6.pt",
                   help="Path to trained model weights (.pt).")
    p.add_argument("--params_csv", default="EGNN/example/Best_hp.csv",
                   help="CSV with model hyperparameters.")
    p.add_argument("--idx_individual", type=int, default=6,
                   help="Row index in params CSV (default: 6).")
    p.add_argument("--train_csv_path", default="pyg_datasets_connected/train/raw/train.csv",
                   help="If y_mean/y_std not provided, compute from this CSV (needs ph_optimum).")
    p.add_argument("--y_mean", type=float, default=7.1956, help="Training-set mean of pH optimum.")
    p.add_argument("--y_std",  type=float, default=1.2302, help="Training-set std of pH optimum.")
    p.add_argument("--pqr_dir", default="./pqr_files/", help="Directory for intermediate PQR files.")
    p.add_argument("--save_dir", default="./pred_out/", help="Output directory for predictions/attentions.")
    p.add_argument("--att-export", choices=["none", "edge", "node", "both"], default="node",
                   help="What attention tables to export.")
    p.add_argument("--node-agg", choices=["sum", "mean"], default="mean",
                   help="How to aggregate edge attentions to nodes (if exporting node).")
    p.add_argument("--charge-power", type=int, default=2, help="Charge power for preprocess_input.")
    return p.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.pqr_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Hyperparams & label stats
    quiet_print(f"Loading hyperparameters from {args.params_csv} (row {args.idx_individual}) ...")
    cfg = load_hparams(args.params_csv, args.idx_individual)

    y_mean, y_std = load_y_stats(args.train_csv_path, args.y_mean, args.y_std)
    quiet_print(f"Using y mean/std: {y_mean:.4f} / {y_std:.4f}")

    # charge scale (must match training)
    _, charge_scale_val = qm9_dataset.retrieve_dataloaders(1, 0)
    charge_scale = float(charge_scale_val)

    # Load checkpoint weights (but DO NOT build the model yet)
    quiet_print(f"Loading model weights from: {args.model_weights}")
    state = torch.load(args.model_weights, map_location=DEVICE)

    # Collect files
    if os.path.isdir(args.input_path):
        pdb_files = sorted(glob.glob(os.path.join(args.input_path, "*.pdb")))
    else:
        pdb_files = [args.input_path] if args.input_path.lower().endswith(".pdb") else []

    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found under {args.input_path}")

    model = None  # will be built after we infer dims from the first successful graph
    results = []

    for pdb_path in tqdm(pdb_files, desc="Predicting", ncols=90):
        base = os.path.splitext(os.path.basename(pdb_path))[0]
        pqr_path = os.path.join(args.pqr_dir, base + ".pqr")

        if not os.path.exists(pqr_path):
            try:
                run_pdb2pqr(pdb_path, pqr_path)
            except Exception as e:
                quiet_print(f"[SKIP] {base}: PDB2PQR failed ({e})")
                continue

        try:
            graph, atom_entries = process_structure_to_graph(pdb_path, pqr_path)
        except Exception as e:
            quiet_print(f"[SKIP] {base}: graph build failed ({e})")
            continue

        # Build the model once with correct inferred dims, then load weights
        if model is None:
            in_node_nf, in_edge_nf = infer_in_dims_from_graph(graph, args.charge_power, charge_scale)
            model = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=in_edge_nf,
                hidden_nf=int(cfg["num_node_features"]),
                n_layers=int(cfg["num_layers"]),
                attention=bool(cfg["attention"]),
                device=DEVICE,
                node_attr=0
            ).to(DEVICE)
            model.load_state_dict(state)  # now shapes match
            model.eval()
            # Attach attention hooks (safe even if attention=False)
            register_attention_hooks(model, int(cfg["num_layers"]))

        try:
            pred = predict_one(
                graph=graph,
                atom_entries=atom_entries,
                model=model,
                charge_power=args.charge_power,
                charge_scale=charge_scale,
                y_mean=y_mean,
                y_std=y_std,
                att_export=args.att_export,
                node_agg=args.node_agg,
                save_dir=args.save_dir,    # <<— write directly here
                base_name=base,
            )
            results.append({"pdb": base, "ph_optimum_pred": float(pred)})
        except Exception as e:
            quiet_print(f"[SKIP] {base}: prediction failed ({e})")
            continue

    # Save summary CSV in save_dir
    if results:
        out_csv = os.path.join(args.save_dir, "predictions.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        quiet_print(f"Saved predictions: {out_csv}")
    else:
        quiet_print("No successful predictions produced.")

if __name__ == "__main__":
    main()


