# =============================================================================
# RGCAT — Gal-DINO Radio Galaxy Catalogue Pipeline
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder) – compile CUDA ops for GalDINO
#   Stage 2 (runtime) – lean image with the compiled ops copied in
#
# Build:
#   docker build -t rgcat:latest .
#
# Run (SLURM-less, single tile):
#   docker run --gpus all \
#     -v /path/to/data:/workspace/data \
#     -v /path/to/outputs:/workspace/outputs \
#     rgcat:latest \
#     python catalog_script.py --config config/pipeline.yaml \
#       --tile image.i.EMU_0044-46.SB50180.cont.taylor.0.restored.conv.fits
# =============================================================================

ARG CUDA_VERSION=12.1.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

# ---- Stage 1: builder -------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        python3-venv \
        git \
        build-essential \
        ninja-build \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
 && python -m pip install --upgrade pip

WORKDIR /build

# Install PyTorch first (CUDA 12.1 wheel)
RUN pip install --no-cache-dir \
        torch==2.3.1+cu121 \
        torchvision==0.18.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Copy only what's needed to compile the CUDA ops
COPY galdino/ /build/galdino/

# Compile deformable attention CUDA ops
RUN cd /build/galdino/models/dino/ops \
 && python setup.py build install

# ---- Stage 2: runtime -------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Allow astropy to write its cache inside the container
    XDG_CACHE_HOME=/tmp/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        wcslib-dev \
        libcfitsio-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
 && python -m pip install --upgrade pip

WORKDIR /workspace

# Install PyTorch (runtime only — no nvcc needed)
RUN pip install --no-cache-dir \
        torch==2.3.1+cu121 \
        torchvision==0.18.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Copy compiled CUDA ops from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages/ \
                    /usr/local/lib/python${PYTHON_VERSION}/dist-packages/

# Install GalDINO framework Python deps
COPY galdino/requirements.txt /tmp/galdino_requirements.txt
RUN pip install --no-cache-dir \
        cython \
        submitit \
        addict \
        yapf \
        termcolor \
        timm \
        git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi \
    && pip install --no-cache-dir -r /tmp/galdino_requirements.txt

# Install pipeline deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the full repo into the image
COPY . /workspace/

# Create volume mount points for data (tiles, catalogs) and outputs
# Mount at runtime with  -v /host/data:/workspace/data
RUN mkdir -p /workspace/data /workspace/outputs /workspace/logs

# Non-root user for security
RUN useradd -m -u 1000 rgcat && chown -R rgcat:rgcat /workspace
USER rgcat

# Default: run the orchestrator with a --help so the image is self-documenting
ENTRYPOINT ["python", "catalog_script.py"]
CMD ["--help"]
