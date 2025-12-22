#!/usr/bin/env python3
"""
Bulk runner for map_active_sites_to_pdb.py over a CSV, including pdb_resseq in the master summary.

Usage :
  python run_bulk_mapping_from_csv_with_resseq.py \
    --csv mapping_ready.csv \
    --pdb-dir ./pdbs \
    --outdir ./mapped_results \
    --mapper-path ./map_active_sites_to_pdb.py \
    --id-col uniprot_id \
    --fasta-col fasta_path \
    --sites-col active_site_positions \
    --default-chain A \
    --workers 8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm
import importlib.util


def load_mapper(mapper_path: str):
    spec = importlib.util.spec_from_file_location("mapper_mod", mapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import from {mapper_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod) 
    if not hasattr(mod, "map_one"):
        raise AttributeError("map_active_sites_to_pdb.py missing function 'map_one'")
    return mod


def find_pdbs_for_id(pdb_dir: Path, acc: str, filename_template: Optional[str]) -> List[Path]:
    if filename_template:
        name = filename_template.format(id=acc)
        p = pdb_dir / name
        return [p] if p.exists() and p.is_file() else []
    # heuristic search
    found: List[Path] = []
    acc_l = acc.lower()
    for p in pdb_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".pdb", ".cif", ".mmcif"]:
            continue
        n = p.name.lower()
        if n in (f"{acc_l}.pdb", f"{acc_l}.cif", f"{acc_l}.mmcif"):
            found.append(p); continue
        if n.startswith(acc_l + "_"):
            found.append(p); continue
        tokens = re.split(r"[_.\-]", n)
        if acc_l in tokens:
            found.append(p); continue
    return found


def read_pdb_resseq_list(out_csv_path: Path) -> Dict[str, str]:
    req = []
    resseqs = []
    if not out_csv_path.exists():
        return {"requested_full_pos": "", "pdb_resseq_list": ""}
    try:
        with out_csv_path.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                full_pos = str(row.get("full_pos", "")).strip()
                in_pdb = str(row.get("in_pdb", "")).strip().lower()
                pdb_resseq = str(row.get("pdb_resseq", "")).strip()
                if full_pos:
                    req.append(full_pos)
                if in_pdb == "yes" and pdb_resseq != "":
                    resseqs.append(pdb_resseq)
        return {
            "requested_full_pos": ",".join(req),
            "pdb_resseq_list": ",".join(resseqs),
        }
    except Exception:
        return {"requested_full_pos": ",".join(req), "pdb_resseq_list": ",".join(resseqs)}


def worker(task):
    mapper_path = task["mapper_path"]
    pdb_path = Path(task["pdb_path"])
    chain = task["chain"]
    fasta_path = Path(task["fasta_path"])
    sites_spec = task["sites_spec"]
    out_csv = Path(task["out_csv"])

    try:
        mod = load_mapper(mapper_path)
        summary = mod.map_one(pdb_path, chain, fasta_path, sites_spec, out_csv)
        extra = read_pdb_resseq_list(out_csv)
        return {
            "status": "OK",
            "pdb_file": pdb_path.name,
            "chain": chain,
            "uniprot_fasta": str(fasta_path),
            "sites_spec": sites_spec,
            "out_csv": out_csv.name,
            **summary,
            **extra,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "pdb_file": pdb_path.name,
            "chain": chain,
            "uniprot_fasta": str(fasta_path),
            "sites_spec": sites_spec,
            "out_csv": out_csv.name,
            "error": str(e),
            "requested_full_pos": "",
            "pdb_resseq_list": "",
        }


def main():
    ap = argparse.ArgumentParser(description="Bulk runner for map_active_sites_to_pdb.py using a CSV + PDB directory, collecting pdb_resseq into summary.")
    ap.add_argument("--csv", required=True, help="Input CSV with UniProt IDs, FASTA paths, and site specs.")
    ap.add_argument("--pdb-dir", required=True, help="Directory containing PDB/mmCIF files.")
    ap.add_argument("--outdir", required=True, help="Directory to write mapping CSVs.")
    ap.add_argument("--mapper-path", required=True, help="Path to map_active_sites_to_pdb.py.")
    ap.add_argument("--id-col", default="uniprot_id", help="Column with UniProt accession.")
    ap.add_argument("--fasta-col", default="fasta_path", help="Column with local FASTA path.")
    ap.add_argument("--sites-col", default="active_site_positions", help="Column with comma-separated active-site positions.")
    ap.add_argument("--chain-col", default="chain", help="Optional column with chain ID; used if present.")
    ap.add_argument("--default-chain", default="A", help="Fallback chain ID if chain-col is missing/empty.")
    ap.add_argument("--filename-template", default=None, help="Optional filename template for PDBs, e.g., '{id}.pdb' or '{id}_A.cif'.")
    ap.add_argument("--workers", type=int, default=4, help="Process parallelism.")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    for col in [args.id_col, args.fasta_col, args.sites_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing column '{col}' in {args.csv}")

    tasks = []
    for _, row in df.iterrows():
        acc = str(row[args.id_col]).strip()
        fasta_path = str(row[args.fasta_col]).strip()
        sites_spec = str(row[args.sites_col]).strip()
        chain = ""
        if args.chain_col in df.columns:
            chain = str(row.get(args.chain_col, "")).strip()
        if not chain:
            chain = args.default_chain

        if not sites_spec:
            continue
        if not fasta_path or not os.path.isfile(fasta_path):
            continue

        pdbs = find_pdbs_for_id(pdb_dir, acc, args.filename_template)
        if not pdbs:
            continue

        for pdb_path in pdbs:
            out_csv = outdir / f"{pdb_path.stem}_{chain}_site_map.csv"
            tasks.append({
                "mapper_path": args.mapper_path,
                "pdb_path": str(pdb_path),
                "chain": chain,
                "fasta_path": fasta_path,
                "sites_spec": sites_spec,
                "out_csv": str(out_csv),
            })

    if not tasks:
        print("No tasks to run. Check your CSV columns, PDB directory, and filename matching.", file=sys.stderr)
        sys.exit(1)

    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Mapping", unit="job"):
            summaries.append(fut.result())

    master_csv = outdir / "bulk_mapping_summary.csv"
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())
    fieldnames = [k for k in [
        "status","pdb_file","chain","uniprot_fasta","sites_spec","out_csv",
        "pdb_seq_len","full_seq_len","alignment_score","n_sites_requested","n_sites_in_pdb",
        "requested_full_pos","pdb_resseq_list","error"
    ] if k in all_keys]

    with master_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, "") for k in fieldnames})

    print(f"\nDone. Wrote per-PDB maps to: {outdir}")
    print(f"Master summary (with pdb_resseq_list): {master_csv}")
    print(f"Total jobs: {len(summaries)} | OK: {sum(1 for s in summaries if s.get('status')=='OK')} | ERROR: {sum(1 for s in summaries if s.get('status')=='ERROR')}")


if __name__ == "__main__":
    main()
