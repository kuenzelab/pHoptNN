#!/usr/bin/env python3
"""
create_pyg_dataset.py — Build PyTorch Geometric datasets from enzyme structures
-------------------------------------------------------------------------------

• Reads training/test CSVs (with columns incl. `uniprot_id`, `ph_optimum`).
• For each UniProt ID, loads a corresponding MMCIF (*.cif) and PQR (*.pqr).
• Constructs per‑atom node features and edges, with 5‑dim edge attributes
  (RDKit bond features when available; otherwise distance‑binned fallback).
• Saves per‑sample PyG `Data` files and two batched containers for EGNN training.

Key design notes
- Robust RDKit handling: silently falls back to distance bins if RDKit/sanitization fails.
- Clean logging (no prints) and clear CLI interface.
- Optional spherical harmonics node encoding via `--l_max`.

Dependencies
- Biopython (Bio.PDB): parses MMCIF and writes temporary PDB for RDKit.
- RDKit (optional): only used for nicer edge features; safe fallback otherwise.
- torch, torch_geometric, numpy, pandas, tqdm.

"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from Bio.PDB import MMCIFParser, PDBIO  # type: ignore

# Project utilities
from constants import ALL_ATOM_LABELS  # atom name vocabulary
from utils import (
    compute_spherical_harmonics,  # optional SH node encoding
    extract_pqr_data,             # parse PQR to atom list
    build_edges_blockwise,        # build chemically sensible edges
)

# ---------------------------- Logging setup ----------------------------
LOGGER = logging.getLogger("create_pyg_dataset")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


# ---------------------------- RDKit features ---------------------------
try:
    from rdkit import Chem  # type: ignore
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False


def _distance_bin_feature(d: float) -> List[float]:
    """4 distance bins + 1 ring flag placeholder (0.0) → 5-dim vector.

    Bins are chosen to mirror training/prediction scripts:
      [0–1.6), [1.6–3.0), [3.0–5.0), [5.0, ∞)
    """
    if d < 1.6:
        bins = [1.0, 0.0, 0.0, 0.0]
    elif d < 3.0:
        bins = [0.0, 1.0, 0.0, 0.0]
    elif d < 5.0:
        bins = [0.0, 0.0, 1.0, 0.0]
    else:
        bins = [0.0, 0.0, 0.0, 1.0]
    return bins + [0.0]  # ring flag unknown in fallback


def get_edge_features_from_rdkit(cif_file_path: str) -> Dict[Tuple[int, int], List[float]]:
    """Extract 5‑dim edge features from MMCIF via RDKit.

    Returns a mapping {(i,j): [one‑hot SINGLE, DOUBLE, TRIPLE, AROMATIC, in_ring]}
    using 0‑based atom indices. If RDKit is unavailable or sanitization fails,
    returns an empty dict (the caller should fall back to distance bins).
    """
    if not _HAS_RDKIT:
        return {}

    parser = MMCIFParser(QUIET=True)
    temp_pdb_file: Optional[str] = None

    try:
        # Convert CIF → PDB using Biopython (more robust for RDKit)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            temp_pdb_file = tmp.name

        structure = parser.get_structure("tmp", cif_file_path)
        io = PDBIO(); io.set_structure(structure); io.save(temp_pdb_file)

        mol = Chem.MolFromPDBFile(temp_pdb_file, removeHs=False, sanitize=False)
        if mol is None:
            return {}

        # Sanitize (can raise due to valence/formal charges)
        Chem.SanitizeMol(mol)

        bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]

        feats: Dict[Tuple[int, int], List[float]] = {}
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            bt = b.GetBondType()
            vec = [1.0 if bt == t else 0.0 for t in bond_types] + [1.0 if b.IsInRing() else 0.0]
            feats[tuple(sorted((i, j)))] = vec
        return feats

    except Exception as e:
        uniprot_id = os.path.basename(cif_file_path).replace(".cif", "")
        LOGGER.debug("RDKit feature extraction failed for %s: %s", uniprot_id, e)
        return {}

    finally:
        if temp_pdb_file and os.path.exists(temp_pdb_file):
            try:
                os.remove(temp_pdb_file)
            except Exception:
                pass


# ------------------------------ Datasets -------------------------------
@dataclass
class EnzymePaths:
    """Bundle paths for a single enzyme (by UniProt)."""
    uniprot_id: str
    cif_path: str
    pqr_path: str


class EnzymeDataset(Dataset):
    """Per‑enzyme dataset producing a PyG `Data` object per UniProt ID.

    Node features (default): [x,y,z] + charge + one‑hot(ALL_ATOM_LABELS)
    Optional node encoding (when `l_max >= 0`): spherical harmonics Y_lm
    replacing xyz (keeps charge + one‑hot).

    Edge attributes (E×5): RDKit bond features when available; otherwise
    4 distance bins + ring flag placeholder.
    """

    def __init__(
        self,
        root: str,
        filename: str,
        cif_dir: str,
        pqr_dir: str,
        y_mean: float,
        y_std: float,
        *,
        test: bool = False,
        transform=None,
        pre_transform=None,
        norm_y: bool = True,
        l_max: int = -1,
        less_aa: bool = False,
    ) -> None:
        self.cif_dir = cif_dir
        self.pqr_dir = pqr_dir
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.test = test
        self.filename = filename
        self.norm_y = norm_y
        self.l_max = int(l_max)
        self.less_aa = bool(less_aa)
        super().__init__(root, transform, pre_transform)

    # ---------------------- Required PyG properties ---------------------
    @property
    def raw_file_names(self) -> str:
        return self.filename

    @property
    def processed_file_names(self) -> List[str]:
        try:
            self.data = pd.read_csv(self.raw_paths[0])
            if "index" not in self.data.columns:
                self.data.index = range(len(self.data))
        except FileNotFoundError:
            return []
        prefix = "data_test_" if self.test else "data_"
        return [f"{prefix}{i}.pt" for i in self.data.index]

    def download(self) -> None:  # noqa: D401 (PyG hook; no remote download used)
        """No-op (files are local)."""
        return

    # ---------------------------- Processing ---------------------------
    def process(self) -> None:
        self.data = pd.read_csv(self.raw_paths[0])
        desc = f"Processing {'Test' if self.test else 'Train'} Data"
        for index, row in tqdm(self.data.iterrows(), total=len(self.data), desc=desc):
            out_name = f"data_test_{index}.pt" if self.test else f"data_{index}.pt"
            out_path = os.path.join(self.processed_dir, out_name)
            if os.path.exists(out_path):
                continue

            uniprot_id = str(row["uniprot_id"])  # expected in CSV
            cif_path = os.path.join(self.cif_dir, f"{uniprot_id}.cif")
            pqr_path = os.path.join(self.pqr_dir, f"{uniprot_id}.pqr")

            # label
            y = torch.tensor([row["ph_optimum"]], dtype=torch.float32)
            if torch.isnan(y).any():
                LOGGER.warning("Skipping %s: NaN label.", uniprot_id)
                continue
            if self.norm_y:
                y = (y - self.y_mean) / (self.y_std + 1e-8)

            # features
            data = self._build_data(cif_path, pqr_path, y)
            if data is None:
                LOGGER.warning("Skipping %s: failed to build features.", uniprot_id)
                continue

            torch.save(data, out_path)

    # --------------------------- Core builder --------------------------
    def _build_data(self, cif_path: str, pqr_path: str, y: torch.Tensor) -> Optional[Data]:
        # Edge features (RDKit → optional)
        rdkit_edge_feats = get_edge_features_from_rdkit(cif_path)

        # Atom entries from PQR (robust parser lives in utils.py)
        try:
            atom_entries = extract_pqr_data(pqr_path, less_aa=self.less_aa)
        except Exception as e:
            LOGGER.debug("PQR parse failed for %s: %s", os.path.basename(pqr_path), e)
            return None
        if not atom_entries:
            return None

        # Build node tensors
        pos, labels, charges = [], [], []
        for a in atom_entries:
            pos.append([a["x"], a["y"], a["z"]])
            labels.append(a["atom_label"])
            charges.append(a["charge"])

        pos_np = np.asarray(pos, dtype=np.float32)
        pos_t = torch.tensor(pos_np, dtype=torch.float32)
        chg_t = torch.tensor(charges, dtype=torch.float32).unsqueeze(1)
        one_hot = torch.tensor(
            [[1 if lab == L else 0 for L in ALL_ATOM_LABELS] for lab in labels],
            dtype=torch.float32,
        )

        if self.l_max >= 0:
            # Replace xyz with spherical harmonics Y_lm (utils.compute_spherical_harmonics)
            try:
                Y = compute_spherical_harmonics(self.l_max, pos_np)  # -> torch.Tensor [N, ?]
                Y_t = torch.tensor(np.asarray(Y), dtype=torch.float32)
                x = torch.cat([Y_t, chg_t, one_hot], dim=1)
            except Exception as e:
                LOGGER.warning("Spherical harmonics failed (l_max=%d); falling back to xyz. Err=%s", self.l_max, e)
                x = torch.cat([pos_t, chg_t, one_hot], dim=1)
        else:
            x = torch.cat([pos_t, chg_t, one_hot], dim=1)

        # Edges and attributes
        edge_index = build_edges_blockwise(atom_entries)  # [2, E]

        edge_attr_list: List[List[float]] = []
        for e in range(edge_index.size(1)):
            u, v = edge_index[:, e].tolist()
            key = tuple(sorted((u, v)))
            feat = rdkit_edge_feats.get(key)
            if feat is None:
                d = float(torch.norm(pos_t[u] - pos_t[v]).item())
                feat = _distance_bin_feature(d)
            edge_attr_list.append(feat)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # ------------------------------ PyG I/O -----------------------------
    def _get_labels(self, label: float) -> torch.Tensor:
        return torch.tensor([label], dtype=torch.float32)

    def len(self) -> int:  # noqa: D401 (PyG API)
        return self.data.shape[0]

    def get(self, idx: int) -> Data:  # noqa: D401 (PyG API)
        name = f"data_test_{idx}.pt" if self.test else f"data_{idx}.pt"
        return torch.load(os.path.join(self.processed_dir, name), weights_only=False)


class EGNNBatchDataset(Dataset):
    """Groups individual `Data` samples into fixed‑size EGNN mini‑batches.

    The object stores a list of lists (chunks) of `Data`. Each `__getitem__`
    returns a *padded* dictionary of tensors suitable for the training
    code in this repository.
    """

    def __init__(self, grouped_dataset: List[List[Data]], batch_size_egnn: int) -> None:
        self.grouped_dataset = grouped_dataset
        self.batch_size_egnn = int(batch_size_egnn)

    def __len__(self) -> int:
        return len(self.grouped_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = self.grouped_dataset[idx]
        if not batch:
            return {}

        num_atoms = torch.tensor([d.x.size(0) for d in batch], dtype=torch.long)
        max_n = int(num_atoms.max().item())

        node_feat_dim = batch[0].x.size(1)
        edge_feat_dim = batch[0].edge_attr.size(1) if hasattr(batch[0], "edge_attr") else 0

        positions = torch.zeros((self.batch_size_egnn, max_n, 3), dtype=torch.float32)
        charges   = torch.zeros((self.batch_size_egnn, max_n), dtype=torch.float32)
        one_hot   = torch.zeros((self.batch_size_egnn, max_n, node_feat_dim - 4), dtype=torch.bool)
        atom_mask = torch.zeros((self.batch_size_egnn, max_n), dtype=torch.bool)
        ph_opti   = torch.cat([d.y for d in batch])

        edges_list: List[torch.Tensor] = []
        edge_attr_list: List[torch.Tensor] = []
        node_offset = 0

        for i, d in enumerate(batch):
            n = d.x.size(0)
            positions[i, :n] = d.x[:, :3]
            charges[i, :n]   = d.x[:, 3]
            one_hot[i, :n]   = d.x[:, 4:] != 0.0
            atom_mask[i, :n] = True

            if d.edge_index.numel() > 0:
                edges_list.append(d.edge_index + node_offset)
                if hasattr(d, "edge_attr") and d.edge_attr is not None:
                    edge_attr_list.append(d.edge_attr)
            node_offset += n

        edges = torch.cat(edges_list, dim=1) if edges_list else torch.empty((2, 0), dtype=torch.long)
        eattr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, edge_feat_dim), dtype=torch.float32)
        edge_mask = torch.ones(edges.size(1), 1, dtype=torch.float32)

        return {
            "positions": positions,
            "atom_mask": atom_mask,
            "edge_mask": edge_mask,
            "one_hot": one_hot,
            "charges": charges,
            "ph_opti": ph_opti,
            "edges": edges,
            "edge_attr": eattr,
        }

    # Simple serialization helpers for the batched container
    @staticmethod
    def save(path: str, dataset_obj: "EGNNBatchDataset") -> None:
        torch.save(dataset_obj, path)

    @staticmethod
    def load(path: str) -> "EGNNBatchDataset":
        return torch.load(path)


# --------------------------------- CLI ---------------------------------

def _ensure_raw_layout(root_dir: str) -> Tuple[str, str]:
    """Ensure train/test CSVs live in `<root>/train/raw` and `<root>/test/raw`.
    Returns `(train_csv_path, test_csv_path)` within the raw subfolders.
    """
    root_train = os.path.join(root_dir, "train")
    root_test  = os.path.join(root_dir, "test")
    os.makedirs(os.path.join(root_train, "raw"), exist_ok=True)
    os.makedirs(os.path.join(root_test, "raw"), exist_ok=True)

    src_train = os.path.join(root_dir, "train.csv")
    src_test  = os.path.join(root_dir, "test.csv")
    dst_train = os.path.join(root_train, "raw", "train.csv")
    dst_test  = os.path.join(root_test,  "raw", "test.csv")

    if os.path.exists(src_train) and not os.path.exists(dst_train):
        shutil.move(src_train, dst_train)
    if os.path.exists(src_test) and not os.path.exists(dst_test):
        shutil.move(src_test, dst_test)

    return dst_train, dst_test


def main() -> None:
    ap = argparse.ArgumentParser(description="Process enzyme data and create PyTorch Geometric datasets.")
    ap.add_argument("--cif_dir", required=True, help="Directory containing *.cif files (per UniProt id).")
    ap.add_argument("--pqr_dir", required=True, help="Directory containing *.pqr files (per UniProt id).")
    ap.add_argument("--root_dir", required=True, help="Root directory with train.csv/test.csv (or nested in train/raw, test/raw).")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size used to group samples into EGNNBatchDataset.")
    ap.add_argument("--l_max", type=int, default=-1, help="If >=0, replace xyz with spherical harmonics up to l_max.")
    ap.add_argument("--no_norm_y", action="store_true", help="Disable normalization of pH labels.")
    args = ap.parse_args()

    train_csv_path, test_csv_path = _ensure_raw_layout(args.root_dir)
    if not (os.path.exists(train_csv_path) and os.path.exists(test_csv_path)):
        LOGGER.error("Could not find train/test CSVs under %s (or its train/raw & test/raw).", args.root_dir)
        raise SystemExit(1)

    LOGGER.info("Loading train labels from %s", train_csv_path)
    df_train = pd.read_csv(train_csv_path)
    if "ph_optimum" not in df_train.columns:
        raise KeyError("train.csv must contain column 'ph_optimum'")
    y_mean = float(df_train["ph_optimum"].mean())
    y_std  = float(df_train["ph_optimum"].std(ddof=0))
    LOGGER.info("Using y_mean=%.4f, y_std=%.4f", y_mean, y_std)

    # Build per‑sample datasets
    train_ds = EnzymeDataset(
        root=os.path.join(args.root_dir, "train"),
        filename="train.csv",
        cif_dir=args.cif_dir,
        pqr_dir=args.pqr_dir,
        y_mean=y_mean,
        y_std=y_std,
        test=False,
        norm_y=not args.no_norm_y,
        l_max=args.l_max,
    )
    test_ds = EnzymeDataset(
        root=os.path.join(args.root_dir, "test"),
        filename="test.csv",
        cif_dir=args.cif_dir,
        pqr_dir=args.pqr_dir,
        y_mean=y_mean,
        y_std=y_std,
        test=True,
        norm_y=not args.no_norm_y,
        l_max=args.l_max,
    )

    # Group into fixed‑size batches for the EGNN training pipeline
    def _group(ds: EnzymeDataset, bs: int) -> List[List[Data]]:
        return [ds[i : i + bs] for i in range(0, len(ds), bs)]

    egnn_train = EGNNBatchDataset(_group(train_ds, args.batch_size), args.batch_size)
    egnn_test  = EGNNBatchDataset(_group(test_ds,  args.batch_size), args.batch_size)

    out_train = os.path.join(args.root_dir, "train", f"egnn_train_dataset_bs_{args.batch_size}.pt")
    out_test  = os.path.join(args.root_dir, "test",  f"egnn_test_dataset_bs_{args.batch_size}.pt")

    LOGGER.info("Saving batched training data to %s", out_train)
    EGNNBatchDataset.save(out_train, egnn_train)

    LOGGER.info("Saving batched test data to %s", out_test)
    EGNNBatchDataset.save(out_test, egnn_test)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
