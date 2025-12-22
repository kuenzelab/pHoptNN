#!/usr/bin/env python3
"""
===============================================================================
attention2pdb.py — Map EGNN attention scores onto PDB B-factors for visualization.

Reads an attention CSV (from your predictor/attention export) and writes a copy of the
input PDB where the B-factor field holds scaled attention scores. Supports matching by
atom serial number or by (chain, resname, resid, atomname). Can optionally aggregate
attention per residue (mean/max) while still writing per-atom B-factors.

Example:
    python attention2pdb.py --pdb 1abc.pdb \
        --attention_csv pred_out/1abc_attention.csv \
        --mode auto --aggregate residue_max --bf_min 0 --bf_max 100

Exit codes:
    0 = success
    2 = CLI/argument errors
    3 = CSV/PDB parsing or mapping errors

AUTHOR = Raj
===============================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _is_atom(rec: str) -> bool:
    return rec.startswith("ATOM  ") or rec.startswith("HETATM")


@dataclass(frozen=True)
class PDBAtom:
    record: str
    serial: str
    name: str
    altloc: str
    resname: str
    chain: str
    resid: str
    icode: str
    occ: str
    bfac: str

    @staticmethod
    def parse(line: str) -> "PDBAtom":
        return PDBAtom(
            record=line[0:6],
            serial=line[6:11].strip(),
            name=line[12:16].strip(),
            altloc=line[16:17],
            resname=line[17:20].strip(),
            chain=line[21:22],
            resid=line[22:26].strip(),
            icode=line[26:27],
            occ=line[54:60],
            bfac=line[60:66],
        )

    def write_with_bfactor(self, line: str, new_b: float, *, force_occ_1: bool = True) -> str:
        if not _is_atom(line):
            return line
        out = list(line)
        bf_str = f"{new_b:6.2f}"
        if force_occ_1:
            occ_str = f"{1.00:6.2f}"
            out[54:60] = list(occ_str)
        out[60:66] = list(bf_str)
        return "".join(out)


_SCORE_PREF_ORDER = [
    "attention_minmax01",
    "attention_raw",
    "attention_score",
    "attention_zscore",
]


def _find_col(df: pd.DataFrame, choices: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for want in choices:
        if want in cols:
            return cols[want]
    return None


def _pick_score_column(df: pd.DataFrame, user_choice: str) -> str:
    cols_lower = [c.lower() for c in df.columns]
    if user_choice.lower() != "auto":
        if user_choice.lower() in cols_lower:
            return df.columns[cols_lower.index(user_choice.lower())]
        _die(3, f"--score_col '{user_choice}' not found. Available: {', '.join(df.columns)}")

    pref = _find_col(df.rename(columns=str.lower), _SCORE_PREF_ORDER)
    if pref:
        return pref
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        _die(3, "No numeric columns found in attention CSV to use as scores.")
    attish = [c for c in numeric if "att" in c.lower()]
    return attish[0] if attish else numeric[0]


def _scale_scores(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        _die(3, "Selected score column contains no finite values.")
    if np.isclose(vmax, vmin):
        return np.full_like(values, (lo + hi) / 2.0, dtype=float)
    z = (values - vmin) / (vmax - vmin)
    return lo + z * (hi - lo)


def _build_serial_map(df: pd.DataFrame, serial_col: Optional[str]) -> Dict[int, float]:
    if not serial_col:
        return {}
    out: Dict[int, float] = {}
    for _, r in df.iterrows():
        try:
            out[int(r[serial_col])] = float(r["__bfactor__"])
        except Exception:
            pass
    return out


def _build_name_map(
    df: pd.DataFrame,
    atom_label_col: Optional[str],
    resname_col: Optional[str],
    resid_col: Optional[str],
    chain_col: Optional[str],
) -> Dict[Tuple[str, ...], float]:
    if not (atom_label_col and resname_col and resid_col):
        return {}
    out: Dict[Tuple[str, ...], float] = {}
    use_chain = chain_col is not None
    for _, r in df.iterrows():
        try:
            key = (
                (str(r[chain_col]).strip(),) if use_chain and pd.notnull(r[chain_col]) else tuple()
            ) + (str(r[resname_col]).strip(), str(r[resid_col]).strip(), str(r[atom_label_col]).strip())
            out[key] = float(r["__bfactor__"])
        except Exception:
            continue
    return out


def _aggregate_residue(df: pd.DataFrame, how: str, resname_col: str, resid_col: str, chain_col: Optional[str]) -> pd.DataFrame:
    keys = [resname_col, resid_col]
    if chain_col:
        keys = [chain_col] + keys
    grouped = df.groupby(keys)["__bfactor__"]
    agg = grouped.mean() if how == "residue_mean" else grouped.max()
    df = df.merge(agg.rename("__bf_res__"), left_on=keys, right_index=True, how="left")
    df["__bfactor__"] = df["__bf_res__"].fillna(df["__bfactor__"])
    df.drop(columns=["__bf_res__"], inplace=True)
    return df


def _match_bfactor(
    lines: List[str],
    serial_map: Dict[int, float],
    name_map: Dict[Tuple[str, ...], float],
    use_serial: bool,
    use_name: bool,
    keep_original_unmapped: bool,
) -> Tuple[List[str], int, int]:
    total = 0
    mapped = 0
    out_lines: List[str] = []

    chain_aware = any(len(k) == 4 for k in name_map.keys())
    for line in lines:
        if not _is_atom(line):
            out_lines.append(line)
            continue
        total += 1
        meta = PDBAtom.parse(line)
        new_b: Optional[float] = None

        if use_serial:
            try:
                s = int(meta.serial)
                if s in serial_map:
                    new_b = serial_map[s]
            except Exception:
                pass

        if new_b is None and use_name and name_map:
            if chain_aware:
                key = (meta.chain, meta.resname, meta.resid, meta.name)
                new_b = name_map.get(key)
            if new_b is None:
                key2 = (meta.resname, meta.resid, meta.name)
                new_b = name_map.get(key2)

        if new_b is not None:
            mapped += 1
            out_lines.append(meta.write_with_bfactor(line, new_b, force_occ_1=True))
        else:
            out_lines.append(line if keep_original_unmapped else meta.write_with_bfactor(line, 0.00, force_occ_1=True))

    return out_lines, mapped, total


def _die(code: int, msg: str) -> "NoReturn":
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map attention CSV onto PDB B-factors for visualization."
    )
    p.add_argument("--pdb", required=True, help="Input PDB to annotate.")
    p.add_argument("--attention_csv", required=True,
                   help="CSV with columns like atom_number/residue_name/residue_id/atom_label "
                        "and an attention score column.")
    p.add_argument("--out_pdb", default=None,
                   help="Output PDB (default: <input>_attention.pdb)")
    p.add_argument("--mode", choices=["auto", "serial", "name"], default="auto",
                   help="Matching mode. auto: try serial then name; serial: atom serial; "
                        "name: (chain?,resname,resid,atom).")
    p.add_argument("--score_col", default="auto",
                   help="Score column to use (default: auto; tries common names, then numeric 'att*').")
    p.add_argument("--aggregate", choices=["none", "residue_mean", "residue_max"], default="residue_max",
                   help="Aggregate scores per residue (written per atom).")
    p.add_argument("--bf_min", type=float, default=0.0, help="Min B-factor after scaling.")
    p.add_argument("--bf_max", type=float, default=100.0, help="Max B-factor after scaling.")
    p.add_argument("--keep_original_unmapped", action="store_true",
                   help="Keep original B-factors for unmatched atoms (default: set to 0.00).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    in_pdb = args.pdb
    out_pdb = args.out_pdb or os.path.splitext(in_pdb)[0] + "_attention.pdb"

    if not os.path.isfile(in_pdb):
        _die(2, f"PDB not found: {in_pdb}")
    if not os.path.isfile(args.attention_csv):
        _die(2, f"Attention CSV not found: {args.attention_csv}")
    if args.bf_max < args.bf_min:
        _die(2, "--bf_max must be ≥ --bf_min")

    df = pd.read_csv(args.attention_csv)
    if df.empty:
        _die(3, "Attention CSV is empty.")

    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    col_serial = next((lower_map[k] for k in ("atom_number", "serial", "atom_serial", "atom_id") if k in lower_map), None)
    col_aname  = next((lower_map[k] for k in ("atom_label", "atom_name", "name") if k in lower_map), None)
    col_rname  = next((lower_map[k] for k in ("residue_name", "res_name", "resname") if k in lower_map), None)
    col_resid  = next((lower_map[k] for k in ("residue_id", "res_id", "resid", "resseq") if k in lower_map), None)
    col_chain  = next((lower_map[k] for k in ("chain", "chain_id", "chainid") if k in lower_map), None)

    score_col = _pick_score_column(df, args.score_col)

    scores = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    df["__bfactor__"] = _scale_scores(scores, args.bf_min, args.bf_max)

    if args.aggregate != "none" and col_rname and col_resid:
        df = _aggregate_residue(df, args.aggregate, col_rname, col_resid, col_chain)

    serial_map = _build_serial_map(df, col_serial)
    name_map   = _build_name_map(df, col_aname, col_rname, col_resid, col_chain)

    use_serial = (args.mode in ("auto", "serial")) and len(serial_map) > 0
    use_name   = (args.mode in ("auto", "name")) and len(name_map) > 0

    if args.mode == "serial" and not use_serial:
        _die(3, "--mode serial requested but CSV lacks a usable atom serial column.")
    if args.mode == "name" and not use_name:
        _die(3, "--mode name requested but CSV lacks (resname,resid,atom[,+chain]) columns.")

    with open(in_pdb, "r") as fh:
        lines = fh.readlines()

    out_lines, mapped, total = _match_bfactor(
        lines, serial_map, name_map, use_serial, use_name, args.keep_original_unmapped
    )

    with open(out_pdb, "w") as fo:
        fo.writelines(out_lines)

    pct = 0.0 if total == 0 else 100.0 * mapped / total
    print(f"[OK] Wrote: {out_pdb}")
    print(f"Mapped atoms: {mapped}/{total} ({pct:.1f}%)")
    if pct < 80.0 and args.mode == "auto":
        print("Note: low match rate. If numbering differs, try --mode name and ensure the CSV "
              "includes 'chain', 'residue_name', 'residue_id', and 'atom_label' columns.")


if __name__ == "__main__":
    main()