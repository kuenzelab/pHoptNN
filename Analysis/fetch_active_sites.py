import os
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm

def create_directory(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def create_session():
    retry_strategy = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=8, pool_maxsize=8)
    session = requests.Session()
    session.headers.update({"User-Agent": "python-script/1.0"})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def get_uniprot_data(session, uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = session.get(url, timeout=20)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def parse_active_sites(data):
    sites = []
    descriptions = []
    features = data.get("features", [])

    for feature in features:
        if feature.get("type", "").lower() == "active site":
            desc = feature.get("description", "Active site")
            location = feature.get("location", {})
            
            position = location.get("start", {}).get("value")
            if not position:
                position = location.get("position", {}).get("value")
            if not position:
                position = location.get("begin", {}).get("value")
            
            if position:
                sites.append(str(position))
                descriptions.append(desc)
                
    return sites, descriptions

def get_fasta(session, uniprot_id, output_dir):
    create_directory(output_dir)
    filepath = os.path.join(output_dir, f"{uniprot_id}.fasta")

    if os.path.isfile(filepath):
        try:
            with open(filepath, "r") as f:
                content = f.read()
            if content.strip().startswith(">"):
                lines = content.splitlines()
                sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
                if sequence:
                    return filepath, sequence
        except Exception:
            pass

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = session.get(url, timeout=20)
        if response.status_code == 200:
            text = response.text.strip()
            if text.startswith(">"):
                with open(filepath, "w") as f:
                    f.write(text + "\n")
                
                lines = text.splitlines()
                sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
                if sequence:
                    return filepath, sequence
    except Exception:
        pass

    return None, None

def process_protein(row, id_col, fasta_dir, include_cols):
    uid = str(row[id_col]).strip()
    session = create_session()

    json_data = get_uniprot_data(session, uid)
    if not json_data:
        return None

    site_positions, site_types = parse_active_sites(json_data)
    
    if not site_positions:
        return None

    fasta_path, sequence = get_fasta(session, uid, fasta_dir)

    result = {
        "uniprot_id": uid,
        "sequence": sequence or "",
        "sequence_length": len(sequence) if sequence else 0,
        "fasta_path": fasta_path or "",
        "active_site_positions": ",".join(site_positions),
        "active_site_types": ",".join(site_types),
        "n_active_sites": len(site_positions),
    }

    for col in include_cols:
        result[col] = row.get(col, "")
        
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv", required=True)
    parser.add_argument("--out", dest="output_csv", required=True)
    parser.add_argument("--id-col", default="uniprot_id")
    parser.add_argument("--include-cols", nargs="*", default=[])
    parser.add_argument("--fastas-dir", default="fastas")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.id_col not in df.columns:
        print(f"Error: Column {args.id_col} not found in input.")
        return

    extra_columns = [c for c in args.include_cols if c in df.columns]

    fieldnames = [
        "uniprot_id",
        "sequence",
        "sequence_length",
        "fasta_path",
        "active_site_positions",
        "active_site_types",
        "n_active_sites",
    ] + extra_columns

    tasks = [dict(row) for _, row in df.iterrows()]
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_protein, row, args.id_col, args.fastas_dir, extra_columns)
            for row in tasks
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            data = future.result()
            if data:
                results.append(data)

    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    create_directory(output_dir)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Finished. Saved {len(results)} entries to {args.output_csv}")

if __name__ == "__main__":
    main()