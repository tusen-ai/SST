ARG PYTORCH="1.10.0" 
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV_VERSION="1.4.0"
# Change these so they worrk on A40s.
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
#RUN git clone https://github.com/open-mmlab/mmcv.git /mmcv
#WORKDIR /mmcv
#RUN MMCV_WITH_OPS=1 pip install -e . 
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install MMDetection3D
RUN conda clean --all
RUN git clone https://github.com/Jaxang/SST.git /sst
WORKDIR /sst
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
# removed -e

RUN pip install ipdb
RUN pip install numba==0.48

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN conda install cython
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

