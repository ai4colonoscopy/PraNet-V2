# Multi-Class Segmentation - PraNet-V2

### 📌 Overview
Welcome to the **multi-class segmentation** zone of **PraNet-V2**! 🌈🏥  

This directory contains all the goodies for **training, testing, and inference** on **multi-class semantic segmentation** tasks, including **ACDC and Synapse** datasets. 🏆

### 📂 Directory Structure

```
multi-class/
├── EMCAD/            # EMCAD model implementation
├── MERIT/            # MERIT model implementation
├── MIST/             # MIST model implementation
├── model_pth/        # Pre-trained model directory
├── dataset/          # Dataset directory
├── README.md         # This file
```

### 🚀 How to Run

#### Training

```bash
cd ./multi-class/MIST
python -W ignore Synapse_train.py --dual
```

#### Inference

```bash
cd ./multi-class/EMCAD
python -W ignore test_synapse.py --dual
```

#### Evaluation

```bash
# Evaluation results are logged in the `test_log/` directory
```

### 📥 Dataset Download

Please refer to the [main README](../README.md) for dataset download instructions and place the datasets in the multi-class/dataset/ directory.

### 🏆 Citation

If you find our work useful, please consider citing us! 🏆👇

```
@article{hu2025pranet2,
  title={PraNet-V2: Dual-Supervised Reverse Attention for Medical Image Segmentation},
  author={Hu, Bo-Cheng and Ji, Ge-Peng and Shao, Dian and Fan, Deng-Ping},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

