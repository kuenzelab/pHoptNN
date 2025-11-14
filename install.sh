#!/usr/bin/env bash
set -e

ENV_NAME="pHoptNN"

echo ">>> Checking for conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda or Anaconda first."
  exit 1
fi

# Enable 'conda activate' inside this script
eval "$(conda shell.bash hook)"

echo ">>> Creating conda env '$ENV_NAME' with Python 3.10"
conda create -y -n "$ENV_NAME" python=3.10

echo ">>> Activating conda env '$ENV_NAME'"
conda activate "$ENV_NAME"

echo ">>> Installing base packages via conda (from environment.yml)"
conda install -y -c conda-forge -c defaults \
  tqdm=4.66.5 \
  numpy=2.1.1 \
  pandas=2.2.3 \
  scipy=1.14.1 \
  biopython=1.84 \
  pdb2pqr \
  openbabel \
  dssp

echo ">>> Upgrading pip inside the conda env"
pip install --upgrade pip

########################################
#  PyTorch + PyG (via pip, CUDA 12.1)
########################################
echo ">>> Installing PyTorch (CUDA 12.1) via pip"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ">>> Installing PyTorch Geometric stack via pip (CUDA 12.1 wheels)"
pip install torch-geometric==2.6.1 \
            torch-scatter==2.1.2 \
            torch-sparse==0.6.18 \
            torch-cluster==1.6.3 \
            torch-spline-conv==1.2.2 \
            pyg-lib==0.4.0 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

########################################
#  Other pip packages from environment.yml
########################################
echo ">>> Installing remaining pip packages from environment.yml"

pip install \
  aiohappyeyeballs==2.6.1 \
  aiohttp==3.12.14 \
  aiosignal==1.4.0 \
  async-timeout==5.0.1 \
  attrs==25.3.0 \
  click==8.2.1 \
  contourpy==1.3.2 \
  cycler==0.12.1 \
  e3nn==0.5.6 \
  fonttools==4.59.2 \
  freesasa==2.2.1 \
  frozenlist==1.7.0 \
  fsspec==2025.7.0 \
  gemmi==0.7.3 \
  jsonschema==4.25.0 \
  jsonschema-specifications==2025.4.1 \
  kiwisolver==1.4.9 \
  llvmlite==0.45.1 \
  lz4==4.4.4 \
  matplotlib==3.10.6 \
  msgpack==1.1.1 \
  multidict==6.6.3 \
  numba==0.62.1 \
  opt-einsum==3.4.0 \
  opt-einsum-fx==0.1.4 \
  propcache==0.3.2 \
  protobuf==6.31.1 \
  psutil==7.0.0 \
  pyarrow==21.0.0 \
  pynndescent==0.5.13 \
  pyparsing==3.2.3 \
  ray==2.48.0 \
  rdkit==2025.3.3 \
  referencing==0.36.2 \
  rpds-py==0.26.0 \
  scikit-learn==1.7.2 \
  seaborn==0.13.2 \
  tensorboardx==2.6.4 \
  umap-learn==0.5.9.post2 \
  yarl==1.20.1

echo ">>> DONE!"
echo
echo "To use the environment in a new shell, run:"
echo "  conda activate $ENV_NAME"
