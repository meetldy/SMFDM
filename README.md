<TOC>

# Accelerating Super-Resolution via Sparse Momentum and Encoder State Reuse

This repository is an offical implementation of the paper "Accelerating Super-Resolution via Sparse Momentum and Encoder State Reuse".

This repository is still under development.
## Introduction
Image super-resolution aims to reconstruct high-quality images from low-resolution inputs, a persistent challenge in computer vision with critical applications in medical imaging, satellite imagery, and video enhancement. Traditional diffusion-based methods, while effective in restoring fine details, suffer from computational inefficiency due to their iterative denoising process. To address this, we introduce the Sparse Momentum-based Faster Diffusion Model (SMFDM), designed for rapid and high-fidelity super-resolution. SMFDM integrates a novel encoder state reuse mechanism that selectively omits non-critical time steps during the denoising phase, significantly reducing computational redundancy. Additionally, the model employs a sparse momentum mechanism, enabling robust representation capabilities while
utilizing only a fraction of the original model weights. Experiments demonstrate that SMFDM achieves an impressive 80.54\% acceleration in the diffusion process, requiring only 15\% of the original weights, while maintaining high-quality outputs with effective preservation of image details and textures. Our work highlights the potential of combining sparse learning and efficient sampling strategies to enhance the practical applicability of diffusion models for super-resolution tasks.

## Environment configuration

The codes are based on python3.7+, CUDA version 11.0+. The specific configuration steps are as follows:

1. Create conda environment
   
   ```shell
   conda env create -f smfdm.yaml
   conda activate smfdm
   ```
## Data preparation
Firstly, download the datasets used.
- [FFHQ](https://github.com/NVlabs/ffhq-dataset) | [CelebaHQ](https://www.kaggle.com/badasstechie/celebahq-resized-256x256)

Then, resize to get LR_IMGS and HR_IMGS.
```
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 
```
## Pre-trained checkpoints

The pre-trained checkpoints can be found at the following: [link](https://drive.google.com/drive/folders/1VISy9fVWa9iOSr6F4oVtKVTOViWuKohQ?usp=drive_link).
## Training and Validation
Run the following command for the training and validation:

   ```shell
   sh run.sh
   ```
## Acknowledgements
This code is mainly built on [IDM](https://github.com/Ree1s/IDM), [sparse_learning](https://github.com/TimDettmers/sparse_learning), and [Faster-Diffusion
](https://github.com/hutaiHang/Faster-Diffusion).
