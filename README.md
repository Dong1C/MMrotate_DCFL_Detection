# ALL files that is needed to build the MMrotate DCFL Model

**Appreciation for these incredible repos's authors and contributors**

- easy steps for building the environments of running the repositry

## 1. step 1 CUDA(compile the cuda) and install the torch

1. A: check the cuda version and compile the cuda
  just follow the tutorial released by NVIDIA OFFICIAL

2. B: just create a new env and install the torch and cudatoolkit

  ```bash
    conda create -n openmmlab python=3.8
    conda install cudatoolkit torch torchvision ...
  ```

## 2. follow the guide in the MMCV-DCFL to build mmcv

## 3. follow the guide in mmdetection to build mmdet


## 4. follow the guide in mmrotate-dcfl to build mmrotate

## 5. work in the mmrotate-dcfl directory
