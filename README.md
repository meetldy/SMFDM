<TOC>

# Accelerating Super-Resolution via Sparse Momentum and Encoder State Reuse

This repository is an offical implementation of the paper "Accelerating Super-Resolution via Sparse Momentum and Encoder State Reuse".

This repository is still under development.

## Environment configuration

The codes are based on python3.7+, CUDA version 11.0+. The specific configuration steps are as follows:

1. Create conda environment
   
   ```shell
   conda env create -f smfdm.yaml
   conda activate smfdm
   ```
## Data preparation
Firstly, download the prepared training datasets and test datasets used.
- [FFHQ and CelebaHQ](10.5281/zenodo.14957655)

Then, resize to get LR_IMGS and HR_IMGS.
```
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 
```

## Training and Validation
Run the following command for the training and validation:

   ```shell
   sh run.sh
   ```
