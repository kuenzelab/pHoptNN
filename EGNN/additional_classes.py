"""
===============================================================================
additional_classes.py 

- Provides EnzymeDatasetAACoarseGrain, a lightweight PyG Dataset that
  builds coarse‑grained amino‑acid graphs from mmCIF files.
- Each residue becomes one node (centroid of its atoms), features are
  [coords or spherical harmonics] + one‑hot residue type.
- Edges connect consecutive residues within a chain (backbone order).

Usage example
-------------

>>> ds = EnzymeDatasetAACoarseGrain(
...     root="./data/aacg",                # holds raw/ and processed/
...     csv_filename="metadata.csv",       # must live under root/raw/
...     cif_dir="./cif",                   # where <uniprot_id>.cif lives
...     test=False,
...     norm_X=True,                        # project residue centroids to unit sphere
...     norm_y=True,                        # normalize labels with y_mean/y_std
...     l_max=-1,                           # set >=0 to use real SH features
...     y_mean=7.20, y_std=1.23             # required when norm_y=True
... )
>>> ds.process()                            # writes processed/*.pt files

The CSV at root/raw/<csv_filename> must contain at least the columns:
  - uniprot_id  (used to locate <cif_dir>/<uniprot_id>.cif)
  - ph_optimum  (target label)
  
AUTHOR = Raj
===============================================================================

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

# Project utilities
from utils import (
    get_atom_indices_type_aa_type_and_coords,
    project_to_unit_sphere,
    compute_spherical_harmonics,
)

# ------------------------- Constants -------------------------
AA_DICT: Dict[str, str] = {
    "CYS": "C", "ASP": "D", "SER": "S", "GLN": "Q", "LYS": "K",
    "ILE": "I", "PRO": "P", "THR": "T", "PHE": "F", "ASN": "N",
    "GLY": "G", "HIS": "H", "LEU": "L", "ARG": "R", "TRP": "W",
    "ALA": "A", "VAL": "V", "GLU": "E", "TYR": "Y", "MET": "M",
    "UNK": "X",
}
AA_ORDER: List[str] = list(AA_DICT.keys())  # deterministic one‑hot ordering


@dataclass(frozen=True)
class _Row:
    uniprot_id: str
    ph_optimum: float


# -------------------- Dataset (coarse‑grained) --------------------
class EnzymeDatasetAACoarseGrain(Dataset):
    """Coarse‑grain each residue to one node (centroid of atoms).

    Node features
    -------------
    - If l_max >= 0: [Y_lm(centroid_dir) ...] + one‑hot(AA)
    - Else:          [x, y, z] + one‑hot(AA)

    Edges
    -----
    - Undirected edges between consecutive residues within the same chain.

    Labels
    ------
    - Single scalar: pH optimum (optionally normalized).
    """

    def __init__(
        self,
        *,
        root: str,
        csv_filename: str,
        cif_dir: str,
        test: bool = False,
        norm_X: bool = False,
        norm_y: bool = False,
        l_max: int = -1,
        y_mean: float | None = None,
        y_std: float | None = None,
        transform=None,
        pre_transform=None,
    ) -> None:
        self.filename = csv_filename
        self.cif_dir = cif_dir
        self.test = bool(test)
        self.norm_X = bool(norm_X)
        self.norm_y = bool(norm_y)
        self.l_max = int(l_max)
        self.y_mean = float(y_mean) if y_mean is not None else None
        self.y_std = float(y_std) if y_std is not None else None
        super().__init__(root, transform, pre_transform)

        if self.norm_y and (self.y_mean is None or self.y_std is None):
            raise ValueError("norm_y=True requires y_mean and y_std.")

    # --------------- PyG required properties/methods ---------------
    @property
    def raw_file_names(self) -> str:
        return self.filename

    @property
    def processed_file_names(self) -> List[str]:
        df = pd.read_csv(self.raw_paths[0])
        if "index" in df.columns:
            idx = list(df["index"].astype(int))
        else:
            idx = list(range(len(df)))
        prefix = "data_test_" if self.test else "data_"
        return [f"{prefix}{i}.pt" for i in idx]

    def download(self) -> None:  # not used
        return None

    # ---------------------------- Build ----------------------------
    def process(self) -> None:
        df = pd.read_csv(self.raw_paths[0])
        it = tqdm(df.iterrows(), total=len(df), desc=("Test" if self.test else "Train"))
        for index, row in it:
            out_path = os.path.join(
                self.processed_dir,
                f"data_test_{index}.pt" if self.test else f"data_{index}.pt",
            )
            if os.path.exists(out_path):
                continue

            try:
                r = _Row(uniprot_id=str(row["uniprot_id"]), ph_optimum=float(row["ph_optimum"]))
            except Exception as exc:
                print(f"[WARN] Row {index}: missing required columns — {exc}")
                continue

            cif_path = os.path.join(self.cif_dir, f"{r.uniprot_id}.cif")
            if not os.path.exists(cif_path):
                print(f"[WARN] CIF not found for {r.uniprot_id}: {cif_path}")
                continue

            try:
                x, edge_index = self._node_features_and_edges(cif_path)
            except Exception as exc:
                print(f"[WARN] Failed {r.uniprot_id}: {exc}")
                continue

            y = torch.tensor([r.ph_optimum], dtype=torch.float32)
            if self.norm_y:
                y = (y - self.y_mean) / self.y_std  

            data = Data(x=x, edge_index=edge_index, y=y, uniprot_id=r.uniprot_id)
            torch.save(data, out_path)

    # ----------------------- Feature builders ----------------------
    def _node_features_and_edges(self, cif_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (node_features, edge_index) for a single structure.

        We first parse atoms → group them by residue id → centroid per residue.
        """
        _, idx_type_coord, _, _ = get_atom_indices_type_aa_type_and_coords(cif_file)
        if not idx_type_coord:
            raise ValueError("No atoms parsed from CIF.")

        # Group by residue index (idx_res)
        grouped: Dict[int, List[List[object]]] = {}
        for atom_idx, atom_name, idx_res, resname, xyz in idx_type_coord:
            grouped.setdefault(int(idx_res), []).append([atom_idx, atom_name, idx_res, str(resname), tuple(xyz)])

        # Centroids + residue labels
        centroids: List[np.ndarray] = []
        aa_labels: List[str] = []
        for entries in grouped.values():
            coords = np.asarray([e[4] for e in entries], dtype=np.float32)
            centroids.append(coords.mean(axis=0))
            aa_labels.append(str(entries[0][3]))

        positions = np.vstack(centroids).astype(np.float32)
        if self.norm_X:
            positions = project_to_unit_sphere(positions)

        one_hot = self._one_hot_aa(aa_labels)  # (N, |AA|)

        if self.l_max >= 0:
            # Spherical harmonics on unit directions (positions are already centered/normed by project_to_unit_sphere)
            Y = compute_spherical_harmonics(self.l_max, positions)  # torch.Tensor [N, F]
            feats = torch.cat([Y, torch.tensor(one_hot, dtype=torch.float32)], dim=1)
        else:
            feats = torch.tensor(np.hstack([positions, one_hot]), dtype=torch.float32)

        edge_index = self._sequential_edges(num_nodes=feats.size(0))
        return feats, edge_index

    @staticmethod
    def _sequential_edges(num_nodes: int) -> torch.Tensor:
        """Connect i↔i+1 (undirected) for 0..N-2. Shape [2, E]."""
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        src = np.arange(0, num_nodes - 1, dtype=np.int64)
        dst = src + 1
        # make undirected
        edges = np.vstack([np.stack([src, dst], axis=0), np.stack([dst, src], axis=0)])
        return torch.tensor(edges.reshape(2, -1), dtype=torch.long)

    @staticmethod
    def _one_hot_aa(labels: List[str]) -> np.ndarray:
        """One‑hot encode 3‑letter residue names using AA_ORDER.

        Unknown residues map to UNK (X)."""
        idx_map = {aa: i for i, aa in enumerate(AA_ORDER)}
        n = len(labels)
        m = len(AA_ORDER)
        M = np.zeros((n, m), dtype=np.float32)
        for i, lab in enumerate(labels):
            j = idx_map.get(str(lab), idx_map["UNK"])  # default to UNK
            M[i, j] = 1.0
        return M

    # ------------------------- PyG accessors ------------------------
    def len(self) -> int:  # PyG <2.5
        df = pd.read_csv(self.raw_paths[0])
        return len(df)

    def get(self, idx: int) -> Data:  # PyG <2.5
        path = os.path.join(
            self.processed_dir, f"data_test_{idx}.pt" if self.test else f"data_{idx}.pt"
        )
        return torch.load(path, weights_only=False)
