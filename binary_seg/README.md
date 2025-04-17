# Binary Segmentation - PraNet-V2

### 📌 Overview

Welcome to the **binary segmentation** hub of **PraNet-V2**! 🏥✨  

This directory contains everything you need for **training, testing, and inference** on **polyp segmentation** tasks. If you're looking for accurate medical image segmentation, you're in the right place! 💪🎉

### 📂 Directory Structure

```
binary/
├── models/                  # Pre-trained models
├── data/                    # Datasets (see the main README for details)
├── snapshots/               # Checkpoints saved during training
├── MyTrain_med.py           # Training script
├── MyTest_med.py            # Testing/inference script
├── eval.py                  # Evaluation script
├── utils/                   # Utility functions
└── README.md                # This file
```

### 🚀 How to Run

#### **Training**
```bash
python -W ignore MyTrain_med.py --model_type PraNet-V2
```

#### Inference

```bash
python -W ignore MyTest_med.py
```

#### Evaluation

```bash
python -W ignore eval.py
```

### 📥 Dataset Download

To get started, **download the dataset** by following the instructions in the [main README](../README.md) and place it in `binary/data/.`

### 🏆 Citation

If you find our work useful, please consider citing us! 🏆👇

```
@article{hu2025pranet2,
  title={PraNet-V2: Dual-Supervised Reverse Attention for Medical Image Segmentation},
  author={Hu, Bo-Cheng and Ji, Ge-Peng and Shao, Dian and Fan, Deng-Ping},
  journal   = {arXiv preprint arXiv:2504.10986},
  year={2025},
  url       = {https://arxiv.org/abs/2504.10986}
}	
```

