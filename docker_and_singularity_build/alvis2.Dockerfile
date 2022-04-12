FROM nvcr.io/nvidia/pytorch:21.08-py3
# 21.08-py3  
ARG PYTORCH="1.10.0"
ARG CUDA="11.4"
ARG CUDNN="8"
ARG MMCV_VERSION="1.4.0"

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common

WORKDIR /temp/packages
ADD . /temp/packages

RUN add-apt-repository ppa:savoury1/ffmpeg4
RUN apt-get install -y --no-install-recommends pkg-config
RUN apt install -y --no-install-recommends ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev libmagickwand-dev && \
  rm -rf /var/lib/apt/lists/*
# python3-opencv (destroys torch for some reason)
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	 python3 get-pip.py && \
	 rm get-pip.py


# Additions for SST:
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
#RUN git clone -b v1.4.0 https://github.com/open-mmlab/mmcv.git /mmcv
#WORKDIR /mmcv
#RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e .
RUN pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install MMDetection3D
RUN conda clean --all
RUN git clone https://github.com/Jaxang/SST.git /sst
WORKDIR /sst
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir --ignore-installed -e .

RUN pip install ipdb
RUN pip install numba==0.48 --ignore-installed

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN conda install cython
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

