FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
ENV DEBIAN_FRONTEND noninteractive

# https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update 
RUN apt-get install -y python3-opencv \
  ca-certificates \
  python3-dev \
  git \
  wget \
  sudo \
  ninja-build

RUN ln -sv /usr/bin/python3 /usr/bin/python

ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.7/get-pip.py && \
    python3 get-pip.py --user && \
    rm get-pip.py

RUN python3 -m pip install --user \
  opencv-python \
  cmake \
  'git+https://github.com/facebookresearch/fvcore'

ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# https://github.com/facebookresearch/detectron2/releases
RUN python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/PercepTreeV1
