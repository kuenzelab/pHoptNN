# pHoptNN

Predict enzyme **pH optima** directly from 3D protein structures using an **Equivariant Graph Neural Network (EGNN)**.  
Includes dataset generation, training, prediction with attention export, and visualization & analysis tools.

---

## Features

- **EGNN model** with attention and edge features (`egnn.py`)
- **Dataset builder** from CIF/PQR → PyTorch Geometric tensors
- **Training pipeline** (LDS, weighted losses, k-fold)
- **Prediction scripts** for new proteins (auto-align with checkpoint)
- **Attention visualization** (mapped to PDB B-factors)
- **RSA and distance analysis**

---

## Environment Setup

### Conda (recommended)
## This enviouroment supports CUDA => 12.x

```bash
# Create and activate environment
# 1️⃣ Create the environment
 bash install.sh

# 2️⃣ Activate it
conda activate pHoptNN

# 3️⃣ Verify installation
python -m torch.utils.collect_env

```

---

### Usecase:
 
## Use for Prediction:
```bash
python phoptnn_interface.py /path/to/pdb_or_pdb folder --save_dir /path/to/output_folder
```
---

## Example Workflow for Building DataSet, Training and Prediction

```bash
# 1️⃣ Build dataset
python EGNN/create_pyg_dataset.py   --cif_dir /data/cif   --pqr_dir /data/pqr   --root_dir /data/phoptnn_dataset

# 2️⃣ Train EGNN
python EGNN/train.py   --root_dir /data/phoptnn_dataset/train   --df_individuals hp_egnn_grid.csv   --idx_individual 0   --save_models_path runs/checkpoints

# 3️⃣ Predict new structures
python phoptnn_interface.py /path/to/pdb_or_pdb folder --save_dir /path/to/output_folder 
 or more advanced version is:
python EGNN/predict.py   --input_path /data/new_pdbs --pqr_dir /data/pqr  --model_weights runs/checkpoints/model_0.pt   --train_csv_path /data/phoptnn_dataset/train/raw/train.csv   --save_dir pred_out
```

---

## Repository Layout

This table provides a high-level overview of the key directories and scripts in the project.

| Directory / File | Description |
| :--- | :--- |
| **Root** | |
| `README.md` | This file, describing the project and layout. |
| `LICENSE` | The open-source license for the repository. |
| `Dockerfile` | Defines the Docker container for environment replication. |
| `environment.yml` | Conda environment file for setting up dependencies. |
| `phoptnn_interface.py`| Main script or interface for running the PhOptNN model. |
| `repository.txt` | A full, detailed tree of all files (for reference). |
| **Core Model (EGNN)** | |
| `EGNN/` | Main directory for the E(n) Equivariant GNN model. |
| `EGNN/models/egnn.py`| Core EGNN model definition. |
| `EGNN/train.py` | Main script for training the EGNN model (supports LDS, k-fold). |
| `EGNN/predict.py` | Script to run predictions and export attention weights. |
| `EGNN/create_pyg_dataset.py` | Script to build PyTorch Geometric datasets from CIF/PQR files. |
| `EGNN/attention2pdb.py` | Utility to map saved attention weights to PDB B-factors for visualization. |
| `EGNN/rsa_anlysis.py` | Script to analyze the relationship between attention, RSA, and active sites. |
| `EGNN/constants.py` | Project constants (e.g., atom and residue dictionaries). |
| `EGNN/weight/` | Stores trained model weights (`.pt` files). |
| `EGNN/qm9/` | Code related to the QM9 benchmark dataset. |
| **Data** | |
| `pyg_datasets_connected/` | Default directory for processed PyTorch Geometric datasets. |
| `pyg_datasets_connected/train/`| Processed training data. |
| `pyg_datasets_connected/test/` | Processed test data. |
| **Structure Generation (AF3)** | |
| `AF3_jobs/` | Directory for configuring, running, and analyzing AlphaFold 3 jobs. |
| `AF3_jobs/input/` | Contains `.slurm` batch scripts and `.json` inputs for running AF3. |
| `AF3_jobs/input_template/` | Data (MSAs, CIFs) and scripts to generate AF3 input templates. |
| `AF3_jobs/output/` | Example output directory from an AF3 run. |
| **Analysis & Alternative Models** | |
| `Analysis/` | Scripts and notebooks for analyzing model predictions and attention. |
| `Analysis/Glycosidasen_class/` | In-depth analysis for the "Glycosidasen" enzyme class, containing data, PDBs, PQR files, and results. |
| `GCN/` | Implementation of an alternative GCN model. |
| `A0A0A1C3U6.pdb` | An example PDB file, likely for quick testing. |
---

## Dataset Preparation

### Input Data

You’ll need:
- **CIF files**: `/data/cif/{uniprot_id}.cif`
- **PQR files**: `/data/pqr/{uniprot_id}.pqr`
- **Training/Test CSVs**:
  - `train.csv` with columns: `uniprot_id, ph_optimum`
  - `test.csv` with at least `uniprot_id,  ph_optimum`

### Build PyG Dataset

Run:

```bash
python EGNN/create_pyg_dataset.py   --cif_dir /data/cif   --pqr_dir /data/pqr   --root_dir /data/phoptnn_dataset   --batch_size 1   --l_max -1
```

This will generate:

```
/data/phoptnn_dataset/train/egnn_train_dataset_bs_1.pt
/data/phoptnn_dataset/test/egnn_test_dataset_bs_1.pt
```

---

## Training

Train your EGNN model with:

```bash
python EGNN/train.py   --root_dir /data/phoptnn_dataset/train   --df_individuals hp_egnn_grid.csv   --idx_individual 0   --early_stopping_patience 50  --losses_csv_path runs/losses   --save_models_path runs/checkpoints
```

# Notes:
- Supports **Label Distribution Smoothing (LDS)** and weighted loss
- Automatically saves best model checkpoint
- Logs training metrics to CSV

---

### Prediction

## Standard prediction with attention export

```bash
python EGNN/predict.py   --input_path /path/to/pdb_or_folder   --model_weights weight/W_6_attn.pt   --params_csv EGNN/hyperparameters/Best_hp.csv   --idx_individual 6   --train_csv_path /data/phoptnn_dataset/train/raw/train.csv   --y_mean 7.1956 --y_std 1.2302   --pqr_dir ./pqr_files   --save_dir ./pred_out   --att-export node   --node-agg mean   --charge-power 2
```

Outputs:
- `pred_out/predictions.csv`
- `pred_out/{pdb}_attention.csv`
- `pred_out/{pdb}_edge_attention.csv`

---

### Visualizing Attention Analysis

## 1️⃣ Attention → PDB (B-factors)

```bash
python attention2pdb.py   --pdb protein.pdb   --attention_csv pred_out/protein_attention.csv   --out_pdb protein_attention.pdb   --mode auto   --aggregate residue_max
```

Then, open `protein_attention.pdb` in **PyMOL** or **Chimera** and color by B-factor to visualize important residues.

---

## 2️⃣ RSA / Distance Analysis

```bash
python rsa_anlysis.py   --pdb_dir /path/to/pdbs   --att_dir ./pred_out   --out_dir ./rsa_out   --active_csv active_site_map.csv   --dist_bins "0:2.5:10"   --rsa_bins "0:0.05:1.0"
```

Generates CSVs and plots showing how attention relates to surface accessibility and active-site proximity.

---

## Data Expectations

- **Atoms & residues** defined in `constants.py`
- Hydrogens are **excluded** automatically
- Convert pdb to pqr file using "pdb2pqr" with AMBER forcefeild for better performance.  
- Edge features:
  - **5-dim RDKit bond vector + ring flag**
  - Fallback: **distance-bin encoding**

---

### Citation

If you use **pHoptNN** in your research, please cite:


---

### License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

### Acknowledgements

Developed as part of the **pHoptNN** project — integrating EGNN-based geometric learning for enzyme property prediction.

---
