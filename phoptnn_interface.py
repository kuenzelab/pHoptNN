"""
phoptnn_interface.py — production wrapper for pHoptNN inference

Highlights
- Accepts a single .pdb file or a directory of .pdb files.
- Ensures PQR availability (pdb2pqr30 → pdb2pqr → ambpdb) and normalizes ATOM tokenization so your
  PQR parser in utils.extract_pqr_data() reads chain/resid reliably.
- Forwards arguments to your predict.py (attention export, node aggregation, etc.) and can run quietly.

Usage
  # simplest
  python phoptnn_interface.py my_structures/ --quiet

  # with explicit paths
  python phoptnn_interface.py my.pdb \
      --save_dir ./pred_out \
      --pqr_dir ./pqr_files \
      --model_weights ATTbest_model_6/weight/Attention_model_6.pt \
      --params_csv EGNN/example/Best_hp.csv \
      --idx_individual 6 \
      --train_csv_path pyg_datasets_connected/train/raw/train.csv \
      --y_mean 7.1956 --y_std 1.2302 \
      --att-export node --node-agg mean --charge-power 2
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Optional

import warnings
warnings.filterwarnings("ignore")


def which_any(candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        p = shutil.which(name)
        if p:
            return p
    return None


def collect_pdbs(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".pdb":
        return [path]
    if path.is_dir():
        return sorted([p for p in path.iterdir() if p.suffix.lower() == ".pdb"])
    raise FileNotFoundError(f"No PDBs found at {path}")


def ensure_pqr(pdb_path: Path, pqr_dir: Path, pdb2pqr_bin: Optional[str], ambpdb_bin: Optional[str], quiet: bool) -> Path:
    pqr_dir.mkdir(parents=True, exist_ok=True)
    out_pqr = pqr_dir / f"{pdb_path.stem}.pqr"
    if out_pqr.exists():
        return out_pqr

    if pdb2pqr_bin:
        args = [pdb2pqr_bin, "--ff=AMBER", str(pdb_path), str(out_pqr)]
        subprocess.run(args,
                       check=True,
                       stdout=(subprocess.DEVNULL),
                       stderr=(subprocess.DEVNULL), close_fds=True)
        return out_pqr

    if ambpdb_bin:
        with open(pdb_path, "rb") as fin, open(out_pqr, "wb") as fout:
            subprocess.run([ambpdb_bin, "-pqr"],
                           stdin=fin, stdout=fout, check=True,
                           stderr=(subprocess.DEVNULL))
        return out_pqr

    raise RuntimeError("No PQR converter found: need pdb2pqr30/pdb2pqr or ambpdb in PATH.")


def _looks_int(tok: str) -> bool:
    try:
        int(tok); return True
    except Exception:
        return False


def _split_chain_resid(tok: str) -> tuple[Optional[str], Optional[str]]:
    # "A123" → ("A","123"), "123A" → ("", "123") best-effort
    if len(tok) >= 2 and tok[0].isalpha() and tok[1:].isdigit():
        return tok[0], tok[1:]
    digits = "".join(ch for ch in tok if ch.isdigit())
    if digits:
        return "", digits
    return None, None


def normalize_pqr_inplace(pqr_path: Path) -> None:
    """
    Make sure every ATOM line has a dedicated chain token (single letter)
    and numeric residue id right after it. Non-ATOM lines are copied verbatim.
    This keeps utils.extract_pqr_data() happy with 'chain' and 'residue_id'.
    """
    fixes = 0
    with open(pqr_path, "r") as fin, NamedTemporaryFile("w", delete=False, dir=str(pqr_path.parent)) as tmp:
        for line in fin:
            if not line.startswith("ATOM"):
                tmp.write(line)
                continue
            parts = line.strip().split()
            need = False
            if len(parts) >= 6 and len(parts[4]) == 1 and parts[4].isalpha() and _looks_int(parts[5]):
                pass  
            else:
                if len(parts) > 4:
                    tok = parts[4]
                    if _looks_int(tok):
                        parts = parts[:4] + ["A", tok] + parts[5:]
                        need = True
                    else:
                        ch, resid = _split_chain_resid(tok)
                        if resid:
                            parts = parts[:4] + [ch or "A", resid] + parts[5:]
                            need = True
                        else:
                            parts = parts[:4] + ["A"] + parts[4:]
                            need = True
                else:
                    parts = parts + ["A", "1"]
                    need = True
            if need:
                fixes += 1
                line = " ".join(parts) + "\n"
            tmp.write(line)
    if fixes:
        shutil.move(tmp.name, pqr_path)
    else:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def resolve_weights(user_path: Optional[str]) -> Optional[str]:
    if user_path and Path(user_path).exists():
        return user_path
    for cand in ("ATTbest_model_6/weight/Attention_model_6.pt",
                 "EGNN/weight/W_6_attn.pt",
                 "weight/W_6_attn.pt"):
        if Path(cand).exists():
            return cand
    return user_path


def main():
    ap = argparse.ArgumentParser(description="Wrapper for pHoptNN predict.py (quiet PQR + clean args).")
    ap.add_argument("input_path", help="Path to a .pdb or a folder containing .pdb files.")
    ap.add_argument("--save_dir", default="./pred_out", help="Where predictions/attentions will be written.")
    ap.add_argument("--pqr_dir", default="./pqr_files", help="Where temporary PQR files will be created.")
    ap.add_argument("--model_weights", default="EGNN/weight/W_6_attn.pt")
    ap.add_argument("--params_csv", default="EGNN/hyperparameters/Best_hp.csv")
    ap.add_argument("--idx_individual", type=int, default=6)
    ap.add_argument("--train_csv_path", default="pyg_datasets_connected/train/raw/train.csv")
    ap.add_argument("--y_mean", type=float, default=7.1956)
    ap.add_argument("--y_std", type=float, default=1.2302)

    ap.add_argument("--att-export", choices=["none", "edge", "node", "both"], default="node")
    ap.add_argument("--node-agg", choices=["sum", "mean"], default="mean")
    ap.add_argument("--charge-power", type=int, default=2)

    ap.add_argument("--quiet", action="store_true", help="Silence child process and converters.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    input_path = Path(args.input_path).resolve()
    save_dir = Path(args.save_dir).resolve(); save_dir.mkdir(parents=True, exist_ok=True)
    pqr_dir = Path(args.pqr_dir).resolve(); pqr_dir.mkdir(parents=True, exist_ok=True)

    pdb2pqr_bin = which_any(["pdb2pqr30", "pdb2pqr"])
    ambpdb_bin = which_any(["ambpdb"])

    pdbs = collect_pdbs(input_path)

    for pdb in pdbs:
        try:
            pqr = ensure_pqr(pdb, pqr_dir, pdb2pqr_bin, ambpdb_bin, quiet=args.quiet)
            normalize_pqr_inplace(pqr)
        except Exception as e:
            if not args.quiet:
                print(f"[WARN] PQR build failed for {pdb.name}: {e}")

    predict_py = (root / "EGNN/predict.py").resolve()
    if not predict_py.exists():
        sys.exit(f"predict.py not found next to this script at: {predict_py}")

    weights = resolve_weights(args.model_weights) or args.model_weights

    cmd = [
        sys.executable, str(predict_py),
        "--input_path", str(input_path),
        "--model_weights", str(weights),
        "--params_csv", str(args.params_csv),
        "--idx_individual", str(args.idx_individual),
        "--train_csv_path", str(args.train_csv_path),
        "--pqr_dir", str(pqr_dir),
        "--save_dir", str(save_dir),
        "--att-export", str(args.att_export),
        "--node-agg", str(args.node_agg),
        "--charge-power", str(args.charge_power),
    ]
    if args.y_mean is not None:
        cmd += ["--y_mean", str(args.y_mean)]
    if args.y_std is not None:
        cmd += ["--y_std", str(args.y_std)]

    env = os.environ.copy()
    extra_pp = f"{root}:{env.get('PYTHONPATH','')}"
    env["PYTHONPATH"] = extra_pp.strip(":")

    if args.quiet:
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[OK] Predictions written to: {save_dir}")
    else:
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
