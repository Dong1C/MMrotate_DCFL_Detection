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
Check this repos(mainly addition about the deformable-conv and some other important modules)
https://github.com/Chasel-Tsui/MMCV-DCFL.git


## 3. follow the guide in mmdetection to build mmdet
*better version 2.**

## 4. follow the guide in mmrotate-dcfl to build mmrotate
just run the build

## 5. work in the mmrotate-dcfl directory
this is my work routine
