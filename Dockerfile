FROM continuumio/miniconda3

WORKDIR /app

RUN conda create -y -n pHoptNN python=3.10

SHELL ["conda", "run", "-n", "pHoptNN", "/bin/bash", "-c"]

RUN conda install -y -c conda-forge -c defaults \
    tqdm=4.66.5 \
    numpy=2.1.1 \
    pandas=2.2.3 \
    scipy=1.14.1 \
    biopython=1.84 \
    pdb2pqr \
    openbabel \
    dssp \
    freesasa \
    gcc_linux-64 \
    gxx_linux-64 && \
    conda clean -afy

RUN pip install --upgrade pip && \
    pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install torch-geometric==2.6.1 \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-cluster==1.6.3 \
    torch-spline-conv==1.2.2 \
    pyg-lib==0.4.0 \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

RUN pip install \
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

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pHoptNN"]

CMD ["python"]
