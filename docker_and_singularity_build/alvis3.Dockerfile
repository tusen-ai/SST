FROM nvcr.io/nvidia/pytorch:21.08-py3
# 21.08-py3  
ARG PYTORCH="1.10.0"
ARG CUDA="11.4"
ARG CUDNN="8"
ARG MMCV_VERSION="1.4.0"

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	 python3 get-pip.py && \
	 rm get-pip.py

RUN pip install --upgrade pip
# Additions for SST:

# Install MMCV, MMDetection and MMSegmentation
RUN git clone -b v1.4.0 https://github.com/open-mmlab/mmcv.git /mmcv
WORKDIR /mmcv
RUN MMCV_WITH_OPS=1 pip install -e .
#RUN pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install MMDetection3D
RUN conda clean --all
RUN git clone https://github.com/Jaxang/SST.git /sst
WORKDIR /sst
RUN pip install -r requirements/build.txt
RUN rm -rf llvmlite
#<(sed '/^[llvmlite]/d' requirements/build.txt)
RUN pip install --no-cache-dir -e . --ignore-installed llvmlite

RUN pip install ipdb
RUN pip install numba==0.48 --ignore-installed llvmlite
#  --ignore-installed

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN conda install cython
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

