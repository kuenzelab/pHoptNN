#!/usr/bin/env python3
"""
SASA_des.py — Attention vs Distance/SASA using NODE or EDGE attention, with Active-Site CSV
===========================================================================================

- Loads attention tables per protein:
    * NODE attention:  *_attention.csv  (per atom)
    * EDGE attention:  *_edge_attention.csv (per edge) → aggregated to per atom (mean/sum)
- Uses Biopython to parse coordinates (robust; no reliance on FreeSASA coords API).
- Uses FreeSASA to compute per-atom areas → collapsed to per-residue SASA. Computes RSA via Tien et al. maxima.
- Computes per-residue attention (mean of per-atom attention), centroids, and min-distance to active site.
- Active-site CSV must contain: 'uniprot_id' and 'fixed_positions_1based'. Positions can include ranges (e.g., "45-49").
- If a protein has no active-site mapping and --require_active_site is not set, falls back to top-k% attention atoms as pocket.

Outputs in --out_dir:
  - per_residue_physical_input.csv
  - binned_attention_vs_distance_angstrom.csv
  - binned_attention_vs_sasa_A2.csv
  - attention_structural_physical_bins.png
  - percentile_attention_vs_distance.csv
  - percentile_attention_vs_rsa.csv
  - attention_percentile_panelC.png
"""

import os, re, glob, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import freesasa
from Bio.PDB import PDBParser, Selection

# -------------------------- Constants --------------------------
MAX_ASA_TIEN = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
    "UNK": 200.0
}

# -------------------------- Utilities --------------------------
def normcol(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def norm_resname(resn: str) -> str:
    resn = (resn or "").strip().upper()
    table = {"HID":"HIS","HIE":"HIS","HIP":"HIS","ASH":"ASP","GLH":"GLU",
             "CYX":"CYS","CYM":"CYS","MSE":"MET","SEC":"CYS"}
    return table.get(resn, resn)

def parse_range(s: str) -> np.ndarray:
    # "start:step:end" or "x1,x2,..." → numpy array
    if ":" in s:
        a, b, c = s.split(":")
        a, b, c = float(a), float(b), float(c)
        return np.arange(a, c + 1e-9, b, dtype=float)
    return np.array([float(x) for x in s.split(",")], dtype=float)

def list_node_csv(att_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(att_dir, "*_attention.csv")))
    return [f for f in files if not re.search(r"_edge_attention\.csv$", f)]

def list_edge_csv(att_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(att_dir, "*_edge_attention.csv")))

def basename_from_attention(path: str) -> str:
    name = os.path.basename(path)
    name = re.sub(r"_edge_attention\.csv$", "", name)
    name = re.sub(r"_attention\.csv$", "", name)
    name = os.path.splitext(name)[0]
    return name

def pick_attention_column(df: pd.DataFrame,
                          preferred=("attention_norm_baseline1",
                                     "attention_minmax01",
                                     "attention_score")) -> str:
    cols = [c.lower() for c in df.columns]
    for cand in preferred:
        if cand in cols:
            return cand
    for c in df.columns:
        if "att" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No suitable attention column found in CSV.")

def _numeric_resid(resid: str) -> Optional[int]:
    """'123', '123A', ' 45B' → 123 (None if no leading int)."""
    if resid is None: return None
    m = re.match(r"\s*([+-]?\d+)", str(resid))
    return int(m.group(1)) if m else None

# -------------------------- Active-site mapping --------------------------
def load_active_site_map(csv_path: str) -> Dict[str, pd.DataFrame]:
    """
    Expect columns:
      - 'uniprot_id'
      - 'fixed_positions_1based' : e.g., "45, 87; 112-115  240"
    Returns dict: protein_id -> DataFrame with 'resnum_list' (List[int]).
    """
    def parse_positions(s: str) -> List[int]:
        if pd.isna(s): return []
        s = str(s)
        tokens = re.split(r"[,\s;]+", s.strip())
        out = []
        for tok in tokens:
            if not tok: continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                if a.strip().isdigit() and b.strip().isdigit():
                    a_i, b_i = int(a), int(b)
                    lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
                    out.extend(range(lo, hi + 1))
            else:
                if tok.isdigit():
                    out.append(int(tok))
        # unique preserving order
        seen = set(); uniq = []
        for x in out:
            if x not in seen:
                seen.add(x); uniq.append(x)
        return uniq

    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [normcol(c) for c in df.columns]
    if "uniprot_id" not in df.columns or "fixed_positions_1based" not in df.columns:
        raise KeyError("Active-site CSV must contain 'uniprot_id' and 'fixed_positions_1based'.")

    df["uniprot_id"] = df["uniprot_id"].astype(str)
    df["resnum_list"] = df["fixed_positions_1based"].apply(parse_positions)

    out: Dict[str, pd.DataFrame] = {}
    site = df[["uniprot_id","resnum_list"]].rename(columns={"uniprot_id":"protein_id"})
    for pid, sub in site.groupby("protein_id"):
        out[str(pid)] = sub.reset_index(drop=True)
    return out

# -------------------------- Data classes --------------------------
@dataclass
class AtomRec:
    chain: str
    residue_name: str
    residue_id: str   # numeric as string (insertion codes are ignored in centroid step)
    atom_label: str
    x: float
    y: float
    z: float

# -------------------------- Biopython + FreeSASA core --------------------------
def compute_sasa_rsa_and_coords(pdb_path: str):
    """
    Returns:
      atoms_df: per-ATOM coords (chain,residue_name,residue_id,atom_label,x,y,z)
      res_sasa_df: per-RESIDUE (residue_name,residue_id,chain,sasa_residue,rsa)
      res_centroids_df: per-RESIDUE centroids (chain,residue_id,x,y,z)
    Coordinates via Biopython; SASA via FreeSASA collapsed to residues.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path).split(".")[0], pdb_path)

    atom_rows = []
    res_to_coords = {}    # (chain,resid_num) -> [N x 3]
    resname_map  = {}     # (chain,resid_num) -> 3-letter
    for model in structure:
        for chain in model:
            ch = str(chain.id)
            for residue in chain:
                try:
                    if not Selection.is_aa(residue, standard=False):
                        continue
                except Exception:
                    pass
                resid_num = int(residue.id[1])  # numeric part
                resn = norm_resname(residue.get_resname())
                resname_map[(ch, resid_num)] = resn
                coords = []
                for atom in residue.get_atoms():
                    xyz = atom.get_coord().astype(float)
                    coords.append(xyz)
                    atom_rows.append({
                        "chain": ch,
                        "residue_name": resn,
                        "residue_id": str(resid_num),
                        "atom_label": atom.get_name().strip(),
                        "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
                    })
                if coords:
                    res_to_coords[(ch, resid_num)] = np.vstack(coords)

    atoms_df = pd.DataFrame(atom_rows)
    if atoms_df.empty:
        return atoms_df, pd.DataFrame(), pd.DataFrame()

    # FreeSASA residue SASA
    fs_struct = freesasa.Structure(str(pdb_path))
    fs_result = freesasa.calc(fs_struct)

    res_sasa = {}
    n_atoms = int(fs_struct.nAtoms()) if hasattr(fs_struct, "nAtoms") else 0
    get_area = (lambda i: float(fs_result.atomArea(i))) if hasattr(fs_result, "atomArea") else None
    for i in range(n_atoms):
        ch = str(fs_struct.chainLabel(i)) if hasattr(fs_struct, "chainLabel") else ""
        try:
            resi = int(fs_struct.residueNumber(i))
        except Exception:
            continue
        key = (ch, resi)
        res_sasa[key] = res_sasa.get(key, 0.0) + (get_area(i) if get_area else 0.0)

    rows = []
    cent_rows = []
    for (ch, resi), sasa in res_sasa.items():
        resn = resname_map.get((ch, resi), "UNK")
        max_asa = MAX_ASA_TIEN.get(resn, MAX_ASA_TIEN["UNK"])
        rsa = float(sasa) / float(max_asa) if max_asa else np.nan
        rows.append({
            "chain": ch,
            "residue_id": str(resi),
            "residue_name": resn,
            "sasa_residue": float(sasa),
            "rsa": float(rsa)
        })
        if (ch, resi) in res_to_coords:
            ctr = res_to_coords[(ch, resi)].mean(axis=0)
            cent_rows.append({
                "chain": ch, "residue_id": str(resi),
                "x": float(ctr[0]), "y": float(ctr[1]), "z": float(ctr[2])
            })

    res_sasa_df = pd.DataFrame(rows)
    res_centroids_df = pd.DataFrame(cent_rows)

    # Ensure residue_id string type
    for df in (atoms_df, res_sasa_df, res_centroids_df):
        if not df.empty and "residue_id" in df.columns:
            df["residue_id"] = df["residue_id"].astype(str)

    return atoms_df, res_sasa_df, res_centroids_df

# -------------------------- Attention I/O --------------------------
def load_node_attention(att_csv: str) -> pd.DataFrame:
    """Return columns: residue_name,residue_id,atom_label, att_norm (baseline=1 per protein)."""
    df = pd.read_csv(att_csv)
    df.columns = [normcol(c) for c in df.columns]
    if "residue_id" in df.columns: df["residue_id"] = df["residue_id"].astype(str)
    att_col = pick_attention_column(df)
    df["att"] = pd.to_numeric(df[att_col], errors="coerce").fillna(0.0)
    mu = df["att"].mean()
    mu = 1.0 if (not np.isfinite(mu) or np.isclose(mu, 0.0)) else mu
    df["att_norm"] = df["att"] / mu
    return df[["residue_name","residue_id","atom_label","att_norm"]].copy()

def load_edge_attention_and_aggregate(att_csv: str, edge_agg: str = "mean") -> pd.DataFrame:
    """
    Load per-edge attention and aggregate to per-atom 'att_norm'.
    Input must have:
      src_residue_name/src_residue_id/src_atom_label
      dst_residue_name/dst_residue_id/dst_atom_label
      and 'attention_edge' (or any numeric '*att*' column).
    """
    df = pd.read_csv(att_csv)
    df.columns = [normcol(c) for c in df.columns]

    required = [
        ("src_residue_name","src_residue_id","src_atom_label"),
        ("dst_residue_name","dst_residue_id","dst_atom_label")
    ]
    for cols in required:
        for c in cols:
            if c not in df.columns:
                raise KeyError(f"Edge file missing column '{c}' in {os.path.basename(att_csv)}")

    if "attention_edge" not in df.columns:
        att_cols = [c for c in df.columns if "att" in c and pd.api.types.is_numeric_dtype(df[c])]
        if not att_cols:
            raise KeyError("No edge attention column found (expected 'attention_edge').")
        edge_att_col = att_cols[0]
    else:
        edge_att_col = "attention_edge"

    def stack_end(df, prefix):
        block = df[[f"{prefix}_residue_name", f"{prefix}_residue_id", f"{prefix}_atom_label", edge_att_col]].copy()
        block.columns = ["residue_name", "residue_id", "atom_label", "edge_att"]
        return block

    A = pd.concat([stack_end(df, "src"), stack_end(df, "dst")], axis=0, ignore_index=True)
    A["residue_id"] = A["residue_id"].astype(str)

    if edge_agg == "sum":
        per_atom = A.groupby(["residue_name","residue_id","atom_label"], as_index=False)["edge_att"].sum()
    else:
        per_atom = A.groupby(["residue_name","residue_id","atom_label"], as_index=False)["edge_att"].mean()

    per_atom.rename(columns={"edge_att":"att"}, inplace=True)
    mu = per_atom["att"].mean()
    mu = 1.0 if (not np.isfinite(mu) or np.isclose(mu, 0.0)) else mu
    per_atom["att_norm"] = per_atom["att"] / mu
    return per_atom[["residue_name","residue_id","atom_label","att_norm"]].copy()

# -------------------------- Merge & per-residue --------------------------
def merge_attention_with_atoms(att_atom_df: pd.DataFrame, atoms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join attention (per-atom) with Biopython atom coords.
    Keys: residue_name,residue_id,atom_label (chain ignored).
    """
    A = atoms_df.copy()
    A.columns = [normcol(c) for c in A.columns]
    B = att_atom_df.copy()
    B.columns = [normcol(c) for c in B.columns]
    if "residue_id" in B.columns: B["residue_id"] = B["residue_id"].astype(str)
    if "residue_id" in A.columns: A["residue_id"] = A["residue_id"].astype(str)
    keys = ["residue_name","residue_id","atom_label"]
    df = pd.merge(A, B, on=keys, how="inner")
    return df  # chain,residue_name,residue_id,atom_label,x,y,z,att_norm

def per_residue_table(df_atom: pd.DataFrame,
                      res_sasa_df: pd.DataFrame,
                      res_centroids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-residue table with:
      - att_residue: mean(att_norm) over atoms with attention
      - sasa_residue, rsa: from FreeSASA residue collapse
      - x,y,z: residue centroid (Biopython)
    """
    att_res = (df_atom.groupby(["residue_name","residue_id"], as_index=False)
                      .agg(att_residue=("att_norm","mean")))
    res_merge = pd.merge(att_res, res_sasa_df,
                         on=["residue_name","residue_id"], how="left")
    out = pd.merge(res_merge, res_centroids_df,
                   on=["residue_id"], how="left")
    for c in ["sasa_residue","rsa","x","y","z"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# -------------------------- Active-site distance --------------------------
def active_site_coords_for_protein(site_df: pd.DataFrame, atoms_df: pd.DataFrame) -> np.ndarray:
    """
    Return Nx3 coords for active-site atoms by matching numeric residue ids
    from site_df['resnum_list'] to numeric part of atoms_df['residue_id'].
    """
    if site_df is None or "resnum_list" not in site_df.columns or site_df["resnum_list"].empty:
        return np.empty((0,3), dtype=float)

    target = set()
    for lst in site_df["resnum_list"]:
        if isinstance(lst, (list, tuple)): target.update(lst)
    if not target:
        return np.empty((0,3), dtype=float)

    A = atoms_df.copy()
    A.columns = [normcol(c) for c in A.columns]
    A["resid_num"] = A["residue_id"].apply(_numeric_resid)
    sub = A[A["resid_num"].isin(target)][["x","y","z"]].to_numpy()
    return sub if sub.size else np.empty((0,3), dtype=float)

def distance_to_active_or_fallback(df_res: pd.DataFrame,
                                   df_atom_full: pd.DataFrame,
                                   site_rows: Optional[pd.DataFrame],
                                   fallback_topk: float = 10.0,
                                   require_active: bool = False) -> Tuple[pd.DataFrame, bool]:
    """
    Min distance from residue centroids (x,y,z in df_res) to pocket atoms.
    Pocket = active-site atoms if available; otherwise top-k% attention atoms (from df_atom_full if att_norm exists,
    else use all atoms uniformly).
    """
    pocket = None
    used_active = False

    if site_rows is not None and len(site_rows):
        pocket = active_site_coords_for_protein(site_rows, df_atom_full)
        if pocket.size:
            used_active = True

    if pocket is None or pocket.size == 0:
        if require_active:
            return df_res, False
        AA = df_atom_full.copy()
        if "att_norm" not in AA.columns:
            AA["att_norm"] = 1.0
        AA = AA.sort_values("att_norm", ascending=False)
        k = max(1, int(np.ceil(len(AA)*(fallback_topk/100.0))))
        pocket = AA.head(k)[["x","y","z"]].to_numpy()

    R = df_res[["x","y","z"]].to_numpy()
    dmin = [float(np.linalg.norm(pocket - r, axis=1).min()) if pocket.size else np.nan for r in R]
    out = df_res.copy()
    out["dist_to_pocket"] = dmin
    return out, used_active

# -------------------------- Curves & plots --------------------------
def binned_curve(df: pd.DataFrame, xcol: str, attcol: str, bins: np.ndarray) -> pd.DataFrame:
    if "protein_id" not in df.columns:
        raise ValueError("df must contain 'protein_id'.")
    idx = np.digitize(df[xcol].values, bins, right=False)
    valid = (idx > 0) & (idx < len(bins))
    d2 = df.loc[valid, ["protein_id", attcol]].copy()
    d2["bin"] = idx[valid]
    per_prot = d2.groupby(["protein_id","bin"], as_index=False)[attcol].mean()
    agg = per_prot.groupby("bin")[attcol].agg(["mean","std","count"]).reset_index()
    sem = np.where(agg["count"]>1, agg["std"]/np.sqrt(agg["count"]), np.nan)
    centers = 0.5*(bins[1:]+bins[:-1])
    out = pd.DataFrame({
        "x_center": [centers[i-1] for i in agg["bin"]],
        "mean": agg["mean"].astype(float),
        "sem": sem.astype(float),
        "n_proteins": agg["count"].astype(int),
    })
    return out.sort_values("x_center")

def percentile_curve(df: pd.DataFrame, xcol: str, qbreaks: List[int], attcol: str="att_residue") -> pd.DataFrame:
    x = df[xcol].astype(float)
    qs = np.percentile(x.dropna(), qbreaks)
    idx = np.digitize(x, qs, right=False)
    valid = (idx > 0) & (idx < len(qs))
    d2 = df.loc[valid, ["protein_id", attcol]].copy()
    d2["qbin"] = idx[valid]
    per_prot = d2.groupby(["protein_id","qbin"], as_index=False)[attcol].mean()
    agg = per_prot.groupby("qbin")[attcol].agg(["mean","std","count"]).reset_index()
    sem = np.where(agg["count"]>1, agg["std"]/np.sqrt(agg["count"]), np.nan)
    q_centers = 0.5*(qs[1:]+qs[:-1])
    out = pd.DataFrame({
        "percentile_center": [q_centers[i-1] for i in agg["qbin"]],
        "mean": agg["mean"].astype(float),
        "sem": sem.astype(float),
        "n_proteins": agg["count"].astype(int),
    })
    out["percentile"] = np.interp(out["percentile_center"], (qs.min(), qs.max()), (0,100))
    return out.sort_values("percentile")

def plot_physical(distance_df: pd.DataFrame, rsa_df: pd.DataFrame, out_png: str):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # Define colors
    dist_color = "tab:blue"
    sasa_color = "tab:orange" # Variable name is fine, it's just for the color

    # --- Plot 1: Distance (Blue) ---
    ax = axes[0]
    ax.plot(distance_df["x_center"], distance_df["mean"], linewidth=3, color=dist_color)
    ax.fill_between(distance_df["x_center"],
                    distance_df["mean"]-distance_df["sem"],
                    distance_df["mean"]+distance_df["sem"], alpha=0.25, color=dist_color)
    ax.axhline(1.0, ls="--", color="k", lw=1)
    ax.set_xlabel("Distance to active site (Å)")
    ax.set_ylabel("Attention (baseline = 1.0)")
    ax.tick_params(direction="out")

    # --- Plot 2: RSA (Orange) ---
    ax = axes[1]
    ax.plot(rsa_df["x_center"], rsa_df["mean"], linewidth=3, color=sasa_color)
    ax.fill_between(rsa_df["x_center"],
                    rsa_df["mean"]-rsa_df["sem"],
                    rsa_df["mean"]+rsa_df["sem"], alpha=0.25, color=sasa_color)
    ax.axhline(1.0, ls="--", color="k", lw=1)
    ax.set_xlabel("RSA")  # <-- CHANGED LABEL
    ax.set_ylabel("Attention (baseline = 1.0)")
    ax.tick_params(direction="out")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_panelC(dist_pct: pd.DataFrame, rsa_pct: pd.DataFrame, out_png: str):
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.8))

    ax.plot(dist_pct["percentile"], dist_pct["mean"], lw=3, label="Distance to active site")
    ax.fill_between(dist_pct["percentile"],
                    dist_pct["mean"]-dist_pct["sem"],
                    dist_pct["mean"]+dist_pct["sem"], alpha=0.25)

    ax.plot(rsa_pct["percentile"], rsa_pct["mean"], lw=3, label="RSA")
    ax.fill_between(rsa_pct["percentile"],
                    rsa_pct["mean"]-rsa_pct["sem"],
                    rsa_pct["mean"]+rsa_pct["sem"], alpha=0.20)

    ax.axhline(1.0, ls="--", color="k", lw=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Attention (baseline = 1.0)")
    ax.tick_params(direction="out")
    leg = ax.legend(frameon=True); leg.get_frame().set_edgecolor("#ddd")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# -------------------------- Main --------------------------
# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser(description="SASA/Distance vs Attention (node or edge attention) with active-site CSV")
    ap.add_argument("--pdb_dir", required=True, help="Directory with PDB files.")
    ap.add_argument("--att_dir", required=True, help="Directory with *_attention.csv or *_edge_attention.csv (flat).")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--active_csv", required=True, help="Active-site CSV with 'uniprot_id' & 'fixed_positions_1based'.")

    ap.add_argument("--att_level", choices=["auto","node","edge"], default="auto",
                    help="Which attention files to use.")
    ap.add_argument("--edge_agg", choices=["mean","sum"], default="mean",
                    help="Aggregate incident edge attention per atom via mean or sum.")
    ap.add_argument("--require_active_site", action="store_true",
                    help="If set, skip proteins with no active-site mapping (no fallback).")

    ap.add_argument("--topk_percent_atoms", type=float, default=10.0,
                    help="Fallback top-k%% attention atoms used as pocket if active site is missing.")
    ap.add_argument("--dist_bins", default="0:2.5:10", help="Distance bins Å (start:step:end).")
    # --- MODIFIED THIS ARGUMENT ---
    ap.add_argument("--rsa_bins", default="0:0.05:1.0", help="RSA bins (start:step:end).")
    ap.add_argument("--percentiles", default="0,20,40,60,80,100", help="Percentiles for panel-C.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dist_bins = parse_range(args.dist_bins)
    # --- MODIFIED THIS LINE ---
    rsa_bins = parse_range(args.rsa_bins) # Was: sasa_bins
    qbreaks   = [int(x) for x in args.percentiles.split(",")]

    # Choose attention files
    node_files = list_node_csv(args.att_dir)
    edge_files = list_edge_csv(args.att_dir)
    if args.att_level == "node":
        att_files = node_files
        att_mode = "node"
    elif args.att_level == "edge":
        att_files = edge_files
        att_mode = "edge"
    else:  # auto
        att_files = node_files if node_files else edge_files
        att_mode = "node" if node_files else "edge"

    if not att_files:
        raise FileNotFoundError(f"No attention CSVs found in {args.att_dir} (mode={args.att_level}).")

    # Active-site map
    site_map = load_active_site_map(args.active_csv)

    all_res_tables: List[pd.DataFrame] = []
    used_active_count = 0
    processed = 0

    for att_csv in tqdm(att_files, desc="Processing proteins", ncols=90):
        base = basename_from_attention(att_csv)
        pdb_path = os.path.join(args.pdb_dir, f"{base}.pdb")
        if not os.path.exists(pdb_path):
            cands = glob.glob(os.path.join(args.pdb_dir, f"{base}*.pdb"))
            if not cands:
                print(f"[WARN] PDB for {base} not found; skipping.")
                continue
            pdb_path = cands[0]

        # Load attention (node or edge → node)
        try:
            if att_mode == "node":
                att_atom = load_node_attention(att_csv)
            else:
                att_atom = load_edge_attention_and_aggregate(att_csv, edge_agg=args.edge_agg)
        except Exception as e:
            print(f"[WARN] Attention load failed for {base}: {e}; skipping.")
            continue

        # Biopython coords + FreeSASA residue SASA/RSA
        try:
            atoms_df, res_sasa_df, res_centroids_df = compute_sasa_rsa_and_coords(pdb_path)
        except Exception as e:
            print(f"[WARN] Coord/SASA extraction failed for {base}: {e}; skipping.")
            continue
        if atoms_df.empty:
            print(f"[WARN] No atoms parsed for {base}; skipping.")
            continue

        # Merge attention with atom coords
        df_atom = merge_attention_with_atoms(att_atom, atoms_df)
        if df_atom.empty:
            print(f"[WARN] No atom matches after merge for {base}; skipping.")
            continue
        df_atom["protein_id"] = base

        # Per-residue aggregation (attention + SASA/RSA + centroids)
        df_res = per_residue_table(df_atom, res_sasa_df, res_centroids_df)

        # Distances (active-site preferred; fallback to top-k% attention atoms)
        # (Fix for DataFrame truth value)
        site_rows = site_map.get(base)
        if site_rows is None:
            site_rows = site_map.get(base.upper())
        if site_rows is None:
            site_rows = site_map.get(base.lower())
            
        # (Fix for KeyError: 'residue_id')
        df_res, used_active = distance_to_active_or_fallback(
            df_res,
            df_atom.copy(),  # Pass the full df_atom
            site_rows,
            fallback_topk=args.topk_percent_atoms,
            require_active=args.require_active_site
        )

        if args.require_active_site and not used_active:
            print(f"[WARN] No active-site rows for {base}; skipping (require_active_site).")
            continue

        df_res["protein_id"] = base
        all_res_tables.append(df_res)
        used_active_count += int(used_active)
        processed += 1

    if not all_res_tables:
        raise RuntimeError("No proteins processed successfully.")

    RES = pd.concat(all_res_tables, ignore_index=True)

    # Save per-residue consolidated table
    per_residue_csv = os.path.join(args.out_dir, "per_residue_physical_input.csv")
    RES.to_csv(per_residue_csv, index=False)

    # Physical-units curves
    dist_phys = binned_curve(RES, xcol="dist_to_pocket", attcol="att_residue", bins=dist_bins)
    # --- MODIFIED THIS BLOCK ---
    rsa_phys = binned_curve(RES, xcol="rsa",  attcol="att_residue", bins=rsa_bins)
    dist_phys.to_csv(os.path.join(args.out_dir, "binned_attention_vs_distance_angstrom.csv"), index=False)
    rsa_phys.to_csv(os.path.join(args.out_dir, "binned_attention_vs_rsa.csv"), index=False) # Changed filename
    plot_physical(dist_phys, rsa_phys, out_png=os.path.join(args.out_dir, "attention_structural_physical_bins.png"))
    # --- END MODIFIED BLOCK ---

    # Percentile (panel-C)
    dist_pct = percentile_curve(RES, xcol="dist_to_pocket", qbreaks=qbreaks, attcol="att_residue")
    rsa_pct  = percentile_curve(RES, xcol="rsa",            qbreaks=qbreaks, attcol="att_residue")
    dist_pct.to_csv(os.path.join(args.out_dir, "percentile_attention_vs_distance.csv"), index=False)
    rsa_pct.to_csv(os.path.join(args.out_dir, "percentile_attention_vs_rsa.csv"), index=False)
    plot_panelC(dist_pct, rsa_pct, out_png=os.path.join(args.out_dir, "attention_percentile_panelC.png"))

    print(f"\nProcessed proteins: {processed} (used active-site distances for {used_active_count})")
    print(f"Saved outputs in: {args.out_dir}")
    print(" - per_residue_physical_input.csv")
    print(" - binned_attention_vs_distance_angstrom.csv")
    # --- MODIFIED THIS LINE ---
    print(" - binned_attention_vs_rsa.csv")
    print(" - attention_structural_physical_bins.png")
    print(" - percentile_attention_vs_distance.csv")
    print(" - percentile_attention_vs_rsa.csv")
    print(" - attention_percentile_panelC.png")

if __name__ == "__main__":
    main()
