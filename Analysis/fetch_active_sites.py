#!/usr/bin/env python3
"""
Example:
  python fetch_seq_sites_mp.py \
    --in SPlit_by_Ec.csv \
    --out mapping_ready.csv \
    --id-col uniprot_id \
    --include-cols ec_id protease_class ph_optimum \
    --fastas-dir fastas \
    --workers 8
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm


def ensure_dir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def make_session() -> requests.Session:
    """Session with retries; created inside each process."""
    status_forcelist = (429, 500, 502, 503, 504)
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.5,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s = requests.Session()
    s.headers.update({"User-Agent": "mp-uniprot-fetch/1.0"})
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_uniprot_json(session: requests.Session, uniprot_id: str):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            return r.json(), None
        else:
            return None, f"JSON HTTP {r.status_code}"
    except Exception as e:
        return None, f"JSON err: {e}"


def extract_active_sites_like_yours(data: dict):
    """
    EXACTLY your approach (plus safe fallbacks):
      if feat['type'].lower() == 'active site'
      use feat['location']['start']['value'] (or position/begin as fallback)
    Returns (numbers: list[str], types: list[str])
    """
    residue_numbers = []
    residue_types = []
    for feat in data.get("features", []):
        if str(feat.get("type", "")).lower() == "active site":
            desc = feat.get("description", "") or "Active site"
            loc = feat.get("location", {}) or {}
            pos = (loc.get("start", {}) or {}).get("value")
            if not pos:
                pos_alt = (loc.get("position", {}) or {}).get("value")
                if pos_alt:
                    pos = pos_alt
            if not pos:
                begin = (loc.get("begin", {}) or {}).get("value")
                if begin:
                    pos = begin
            if pos:
                residue_numbers.append(str(pos))
                residue_types.append(desc)
    return residue_numbers, residue_types


def fetch_fasta_and_sequence(session: requests.Session, uniprot_id: str, fastas_dir: str):
    """
    Download FASTA to fastas/<ID>.fasta and return (path, sequence, error).
    Reuses existing file when valid.
    """
    ensure_dir(fastas_dir)
    fasta_path = os.path.join(fastas_dir, f"{uniprot_id}.fasta")

    # Reuse
    if os.path.isfile(fasta_path):
        try:
            with open(fasta_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            seq = "".join(ln for ln in lines if not ln.startswith(">"))
            if seq:
                return fasta_path, seq, None
        except Exception:
            pass

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            return None, None, f"FASTA HTTP {r.status_code}"
        text = r.text.strip()
        if not text.startswith(">"):
            return None, None, "FASTA missing header"
        with open(fasta_path, "w") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))
        lines = text.splitlines()
        seq = "".join(ln.strip() for ln in lines[1:] if ln and not ln.startswith(">"))
        if not seq:
            return fasta_path, None, "FASTA empty sequence"
        return fasta_path, seq, None
    except Exception as e:
        return None, None, f"FASTA err: {e}"


def worker(row_dict: dict, id_col: str, fastas_dir: str, include_cols: list):
    """
    Runs in a separate process. Returns either a dict (active sites found) or None.
    """
    uid = str(row_dict[id_col]).strip()
    session = make_session()

    # Fetch JSON and extract active sites
    json_data, json_err = fetch_uniprot_json(session, uid)
    if json_data:
        numbers, types = extract_active_sites_like_yours(json_data)
    else:
        numbers, types = [], []

    # Only keep entries with active sites
    if not numbers:
        return None

    # Fetch FASTA / sequence
    fasta_path, seq, fasta_err = fetch_fasta_and_sequence(session, uid, fastas_dir)

    # Assemble output row
    out = {
        "uniprot_id": uid,
        "sequence": seq or "",
        "sequence_length": len(seq) if seq else "",
        "fasta_path": fasta_path or "",
        "active_site_positions": ",".join(numbers),  # comma-separated
        "active_site_types": ",".join(types),        # comma-separated
        "n_active_sites": len(numbers),
    }
    # Add requested pass-through columns
    for c in include_cols:
        out[c] = row_dict.get(c, "")
    return out


def main():
    ap = argparse.ArgumentParser(description="Multiprocessing UniProt fetch with tqdm; store ONLY entries with active sites.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV path.")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--id-col", default="uniprot_id", help="Column name containing UniProt accession.")
    ap.add_argument("--include-cols", nargs="*", default=[], help="Extra columns to copy over (if present).")
    ap.add_argument("--fastas-dir", default="fastas", help="Directory to save FASTA files.")
    ap.add_argument("--workers", type=int, default=8, help="Number of processes.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.id_col not in df.columns:
        raise SystemExit(f"Column '{args.id_col}' not found in input CSV.")

    include_cols = [c for c in args.include_cols if c in df.columns]

    # Prepare header
    fieldnames = [
        "uniprot_id",
        "sequence",
        "sequence_length",
        "fasta_path",
        "active_site_positions",
        "active_site_types",
        "n_active_sites",
    ] + include_cols

    # Submit tasks
    rows = [dict(r) for _, r in df.iterrows()]
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, r, args.id_col, args.fastas_dir, include_cols) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching UniProt + active sites", unit="protein"):
            res = fut.result()
            if res is not None:  # keep only FOUND
                results.append(res)

    # Write output (only FOUND)
    ensure_dir(os.path.dirname(os.path.abspath(args.out_csv)) or ".")
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\n✅ Done! Wrote {len(results)} proteins with active sites to: {args.out_csv}")
    print("Use with map_active_sites_to_pdb.py as:")
    print("  --uniprot-fasta : fasta_path")
    print('  --sites         : active_site_positions (e.g. "57,102,195")')


if __name__ == "__main__":
    main()
