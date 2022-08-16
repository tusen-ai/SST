ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install MMDetection3D
RUN conda clean --all
WORKDIR ../
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
# removed -e

RUN pip install ipdb
RUN pip install numba==0.48

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN conda install cython
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

