# https://hub.docker.com/layers/nvidia/cuda/12.0.0-base-ubuntu22.04/images/sha256-3b6f49136ec6725b6fcc0fc04f2f7711d3d1e22d0328da2ca73e52dbd37fa4b1
FROM nvidia/cuda:12.0.0-base-ubuntu22.04

LABEL author="Akos Szolgyen, PhD <akos.szolgyend@northwestern.edu>"
LABEL pi="Lee AD Cooper, PhD <lee.cooper@northwestern.edu>"
LABEL project="MuTILs"

# Use bash as the default shell
SHELL ["/bin/bash", "-c"]

# Update and install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    curl \
    wget \
    vim \
    screen \
    tree \
    rsync \
    git \
    ca-certificates \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libcupti-dev \
    libvips-dev \
    memcached && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /home

# Create and activate a virtual environment
RUN python3 -m venv venv

# Upgrade pip and install Python dependencies
RUN /home/venv/bin/pip install --upgrade pip && \
    /home/venv/bin/pip install \
        Cython==3.0.11 \
        scikit-build==0.18.1 \
        cmake==3.31.4 \
        ipython==8.31.0 \
        jupyter==1.1.1 \
        numpy==2.2.2 \
        pandas==2.2.3 \
        SQLAlchemy==2.0.37 \
        scipy==1.15.1 \
        scikit-learn==1.6.1 \
        scikit-image==0.25.0 \
        imageio==2.37.0 \
        pillow==11.1.0 \
        matplotlib==3.10.0 \
        seaborn==0.13.2 \
        torch==2.5.1 \
        torchvision==0.20.1 \
        GitPython==3.1.44 \
        Deprecated==1.2.16 && \
    /home/venv/bin/pip install histomicstk --find-links https://girder.github.io/large_image_wheels && \
    rm -rf ~/.cache/pip

# Clone the MuTILs_Panoptic repository and histolab submodule
RUN git clone https://github.com/szolgyen/MuTILs_Panoptic \
    && cd MuTILs_Panoptic \
    && git submodule update --init --recursive

# Build Cython modules
RUN cd /home/MuTILs_Panoptic/utils/CythonUtils \
    && /home/venv/bin/python3 setup.py build_ext --inplace

# Set PYTHONPATH
ENV PYTHONPATH="/home/MuTILs_Panoptic:/home/MuTILs_Panoptic/histolab/src:$PYTHONPATH"

# Ensure the virtual environment is automatically activated
RUN echo "source /home/venv/bin/activate" >> /root/.bashrc
