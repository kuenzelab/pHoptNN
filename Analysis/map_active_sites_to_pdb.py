#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from Bio import SeqIO, pairwise2
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import protein_letters_3to1

def parse_active_sites(spec: str) -> List[int]:
    """Parse a spec like "45, 123-127, 250" into sorted unique 1-based positions."""
    positions = set()
    spec = spec.strip()
    if not spec:
        return []
    for part in re.split(r"[,\s]+", spec):
        if not part:
            continue
        if "-" in part:
            a,b = part.split("-")
            a = int(a); b = int(b)
            if a > b:
                a,b = b,a
            for x in range(a, b+1):
                positions.add(x)
        else:
            positions.add(int(part))
    return sorted(positions)

def load_uniprot_sequence(fasta_path: Path) -> str:
    recs = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not recs:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")
    return str(recs[0].seq).upper()
def three_to_one_safe(resname: str) -> str:
    return protein_letters_3to1.get(resname.upper(), 'X')

def get_chain_residues(structure, chain_id: str):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                residues = [res for res in chain.get_residues() if res.id[0] == ' ']
                return residues
    raise ValueError(f"Chain '{chain_id}' not found in structure.")

def pdb_chain_sequence_and_index(residues) -> Tuple[str, List[Tuple[int, str]]]:
    seq = []
    idx = []
    for res in residues:
        resname = res.get_resname().strip()
        aa = three_to_one_safe(resname)
        seq.append(aa)
        hetflag, resseq, icode = res.id
        idx.append((resseq, icode))
    return "".join(seq), idx

def read_structure(pdb_path: Path, chain_id: str):
    if pdb_path.suffix.lower() in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    residues = get_chain_residues(structure, chain_id)
    return structure, residues

def align_full_to_pdb(full_seq: str, pdb_seq: str):
    alns = pairwise2.align.globalms(full_seq, pdb_seq, 1.0, -1.0, -10.0, -0.5, one_alignment_only=True)
    if not alns:
        raise RuntimeError("Alignment failed")
    a = alns[0]
    return a.seqA, a.seqB, a.score

def map_full_positions_to_pdb(full_aligned: str, pdb_aligned: str, pdb_index: List[Tuple[int,str]]) -> Dict[int, Optional[Tuple[int, str, str]]]:
    mapping = {}
    full_i = 0
    pdb_i  = 0
    if len(full_aligned) != len(pdb_aligned):
        raise ValueError("Aligned strings have different lengths, cannot map.")
    for col in range(len(full_aligned)):
        f = full_aligned[col]
        p = pdb_aligned[col]
        if f != '-':
            full_i += 1
            if p != '-':
                resseq, icode = pdb_index[pdb_i]
                mapping[full_i] = (resseq, icode, p)
                pdb_i += 1
            else:
                mapping[full_i] = None
        else:
            if p != '-':
                pdb_i += 1
    return mapping

def write_mapping_csv(out_path: Path, chain_id: str, full_seq: str, pdb_seq: str, mapping: Dict[int, Optional[Tuple[int, str, str]]], active_sites: List[int]):
    fieldnames = [
        "full_pos", "full_aa",
        "in_pdb", "pdb_chain", "pdb_resseq", "pdb_icode", "pdb_aa"
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pos in active_sites:
            if pos < 1 or pos > len(full_seq):
                w.writerow({
                    "full_pos": pos,
                    "full_aa": "",
                    "in_pdb": "no",
                    "pdb_chain": chain_id,
                    "pdb_resseq": "",
                    "pdb_icode": "",
                    "pdb_aa": ""
                })
                continue
            full_aa = full_seq[pos-1]
            entry = mapping.get(pos)
            if entry is None:
                w.writerow({
                    "full_pos": pos,
                    "full_aa": full_aa,
                    "in_pdb": "no",
                    "pdb_chain": chain_id,
                    "pdb_resseq": "",
                    "pdb_icode": "",
                    "pdb_aa": ""
                })
            else:
                resseq, icode, pdb_aa = entry
                w.writerow({
                    "full_pos": pos,
                    "full_aa": full_aa,
                    "in_pdb": "yes",
                    "pdb_chain": chain_id,
                    "pdb_resseq": resseq,
                    "pdb_icode": (icode or "").strip() if isinstance(icode, str) else "",
                    "pdb_aa": pdb_aa
                })

def map_one(pdb_path: Path, chain_id: str, uniprot_fasta: Path, sites_spec: str, out_csv: Path):
    full_seq = load_uniprot_sequence(uniprot_fasta)
    _, residues = read_structure(pdb_path, chain_id)
    pdb_seq, pdb_index = pdb_chain_sequence_and_index(residues)
    if not pdb_seq:
        raise ValueError(f"No residues found for chain '{chain_id}' in {pdb_path}")
    full_aln, pdb_aln, score = align_full_to_pdb(full_seq, pdb_seq)
    mapping = map_full_positions_to_pdb(full_aln, pdb_aln, pdb_index)
    active_sites = parse_active_sites(sites_spec)
    write_mapping_csv(out_csv, chain_id, full_seq, pdb_seq, mapping, active_sites)
    return {
        "pdb_seq_len": len(pdb_seq),
        "full_seq_len": len(full_seq),
        "alignment_score": score,
        "n_sites_requested": len(active_sites),
        "n_sites_in_pdb": sum(1 for s in active_sites if mapping.get(s) is not None)
    }

def main():
    ap = argparse.ArgumentParser(description="Map UniProt active-site positions onto a PDB chain via global alignment.")
    ap.add_argument("--pdb", required=True, nargs="+", help="Path(s) to PDB/mmCIF file(s)")
    ap.add_argument("--chain", default="A", help="Chain ID in the PDB to map onto")
    ap.add_argument("--uniprot-fasta", default="fastas", help="FASTA file with the full UniProt sequence")
    ap.add_argument("--sites", required=True, help='Active-site spec like "45, 123-127, 250"')
    ap.add_argument("--outdir", default=".", help="Output directory for CSV(s)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    uniprot_fasta = Path(args.uniprot_fasta)
    if not uniprot_fasta.exists():
        raise SystemExit(f"FASTA not found: {uniprot_fasta}")

    for pdb in args.pdb:
        pdb_path = Path(pdb)
        if not pdb_path.exists():
            print(f"[WARN] PDB not found: {pdb_path}")
            continue
        out_csv = outdir / f"{pdb_path.stem}_{args.chain}_site_map.csv"
        try:
            summary = map_one(pdb_path, args.chain, uniprot_fasta, args.sites, out_csv)
            print(f"[OK] {pdb_path.name} -> {out_csv.name} | aln_score={summary['alignment_score']:.1f} | full_len={summary['full_seq_len']} | pdb_len={summary['pdb_seq_len']} | sites_in_pdb={summary['n_sites_in_pdb']}/{summary['n_sites_requested']}")
        except Exception as e:
            print(f"[ERROR] {pdb_path.name}: {e}")

if __name__ == "__main__":
    main()