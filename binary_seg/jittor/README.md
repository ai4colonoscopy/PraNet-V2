# 🚀 PraNet-V2 on Jittor Framework 🌟



## Usage

🚀 **First**, please refer to the [official Jittor installation guide](https://cg.cs.tsinghua.edu.cn/jittor/download/) to set up your Jittor environment. 🛠️

✨ For **macOS** and **Windows**, we recommend using Docker 🐳 for a smooth setup. If you’re on **Linux**, simply install Jittor with pip.

We have deployed four models of the PraNet series (PraNet-V1,PVT-PraNet-V1, PlaNet-V2, PVT-PraNet-V2) on the Jittor framework and released the [inference code](https://github.com/ai4colonoscopy/PraNet-V2/blob/main/binary/jittor/MyTest.py)! 

✨ Relevant paths have been configured, and you can easily run the commands below to try inference and evaluation. 🎉👾

```bash
cd ./binary/jittor
python MyTest.py # --bs <int> to set batch_size for test
python eval.py
```



## Performance Comparison

The performance may show slight variations due to differences in operator implementations between the two frameworks.

#### PraNet-V1

| dataset      |  Arch   | mDice (%) | mIoU (%) | wFm (%) | S-m (%) | mEm (%) | MAE (\*1e-2) |
| :----------- | :-----: | :-------: | :------: | :-----: | :-----: | :-----: | :----------: |
| CVC-300      | PyTorch |   87.06   |  79.61   |  84.32  |  92.55  |  94.97  |     0.99     |
|              | Jittor  |   87.00   |  79.53   |  84.16  |  92.50  |  94.93  |     0.99     |
| CVC-ClinicDB | PyTorch |   89.84   |  84.83   |  89.63  |  93.67  |  96.22  |     0.94     |
|              | Jittor  |   89.92   |  84.90   |  89.72  |  93.70  |  96.27  |     0.94     |
| Kvasir       | PyTorch |   89.39   |  83.55   |  88.00  |  91.25  |  94.00  |     3.04     |
|              | Jittor  |   89.41   |  83.58   |  88.01  |  91.28  |  94.03  |     3.02     |
| ETIS         | PyTorch |   62.75   |  56.57   |  60.07  |  79.33  |  80.77  |     3.07     |
|              | Jittor  |   62.62   |  56.47   |  60.00  |  79.25  |  80.85  |     3.18     |



#### PVT-PraNet-V1

| dataset      |  Arch   | mDice (%) | mIoU (%) | wFm (%) | S-m (%) | mEm (%) | MAE (\*1e-2) |
| ------------ | :-----: | :-------: | :------: | :-----: | :-----: | :-----: | :----------: |
| CVC-300      | PyTorch |   86.59   |  78.92   |  83.15  |  91.84  |  94.45  |     1.03     |
|              | Jittor  |   86.63   |  78.85   |  83.18  |  91.87  |  94.66  |     1.01     |
| CVC-ClinicDB | PyTorch |   90.96   |  85.42   |  89.90  |  94.34  |  96.49  |     1.02     |
|              | Jittor  |   90.93   |  85.38   |  89.95  |  94.37  |  96.47  |     1.03     |
| Kvasir       | PyTorch |   87.09   |  81.31   |  84.52  |  89.33  |  92.58  |     4.19     |
|              | Jittor  |   87.24   |  81.52   |  84.81  |  89.51  |  92.73  |     4.10     |
| ETIS         | PyTorch |   68.32   |  60.02   |  61.65  |  81.38  |  80.92  |     4.14     |
|              | Jittor  |   68.74   |  60.55   |  62.21  |  81.68  |  81.37  |     3.94     |



#### PraNet-V2

| dataset      |  Arch   | mDice (%) | mIoU (%) | wFm (%) | S-m (%) | mEm (%) | MAE (\*1e-2) |
| ------------ | :-----: | :-------: | :------: | :-----: | :-----: | :-----: | :----------: |
| CVC-300      | PyTorch |   89.83   |  82.66   |  87.79  |  93.70  |  97.47  |     0.59     |
|              | Jittor  |   89.88   |  82.69   |  88.01  |  93.74  |  97.59  |     0.59     |
| CVC-ClinicDB | PyTorch |   92.28   |  87.22   |  91.97  |  94.87  |  97.38  |     0.91     |
|              | Jittor  |   92.32   |  87.31   |  92.12  |  94.95  |  97.37  |     0.92     |
| Kvasir       | PyTorch |   90.70   |  85.29   |  89.59  |  91.70  |  95.07  |     2.35     |
|              | Jittor  |   90.73   |  85.35   |  89.75  |  91.90  |  95.06  |     2.34     |
| ETIS         | PyTorch |   64.05   |  56.54   |  60.43  |  79.41  |  79.74  |     2.08     |
|              | Jittor  |   64.09   |  56.75   |  60.79  |  79.43  |  79.74  |     2.07     |



#### PVT-PraNet-V2

| dataset      |  Arch   | mDice (%) | mIoU (%) | wFm (%) | S-m (%) | mEm (%) | MAE (\*1e-2) |
| ------------ | :-----: | :-------: | :------: | :-----: | :-----: | :-----: | :----------: |
| CVC-300      | PyTorch |   89.89   |  83.11   |  88.48  |  93.96  |  97.04  |     0.73     |
|              | Jittor  |   89.90   |  83.09   |  88.61  |  94.00  |  97.10  |     0.72     |
| CVC-ClinicDB | PyTorch |   93.09   |  88.06   |  92.80  |  94.45  |  98.23  |     0.84     |
|              | Jittor  |   93.10   |  88.03   |  92.91  |  94.57  |  98.20  |     0.85     |
| Kvasir       | PyTorch |   91.52   |  86.12   |  90.39  |  92.50  |  95.64  |     2.33     |
|              | Jittor  |   91.47   |  86.04   |  90.46  |  92.52  |  95.60  |     2.33     |
| ETIS         | PyTorch |   76.35   |  68.72   |  72.96  |  86.50  |  88.26  |     1.45     |
|              | Jittor  |   76.38   |  68.84   |  73.25  |  86.51  |  88.28  |     1.43     |



## Speedup

The Jittor-based code significantly boosts inference efficiency, achieving nearly **4x faster** single-sample inference! 🚀🔥

#### PraNet-V1

| Batch Size | PyTorch |  Jittor  | Speedup |
| ---------- | :-----: | :------: | :-----: |
| 1          | 32 FPS  | 129 FPS  |  4.03x  |
| 4          | 165 FPS | 345 FPS  |  2.09x  |
| 8          | 328 FPS | 724 FPS  |  2.21x  |
| 16         | 642 FPS | 1457 FPS |  2.27x  |

#### PVT-PraNet-V1

| Batch Size | PyTorch |  Jittor  | Speedup |
| :--------- | :-----: | :------: | :-----: |
| 1          | 28 FPS  | 121 FPS  |  4.32x  |
| 4          | 138 FPS | 482 FPS  |  3.49x  |
| 8          | 261 FPS | 907 FPS  |  3.48x  |
| 16         | 541 FPS | 1772 FPS |  3.28x  |

#### PraNet-V2


| Batch Size | PyTorch |  Jittor  | Speedup |
| :--------- | :-----: | :------: | :-----: |
| 1          | 31 FPS  | 143 FPS  |  4.61x  |
| 4          | 143 FPS | 518 FPS  |  3.62x  |
| 8          | 283 FPS | 986 FPS  |  3.48x  |
| 16         | 556 FPS | 1911 FPS |  3.44x  |

#### PVT-PraNet-V2


| Batch Size | PyTorch |  Jittor  | Speedup |
| ---------- | :-----: | :------: | :-----: |
| 1          | 29 FPS  | 117 FPS  |  4.03x  |
| 4          | 130 FPS | 430 FPS  |  3.31x  |
| 8          | 241 FPS | 777 FPS  |  3.22x  |
| 16         | 493 FPS | 1697 FPS |  3.44x  |



## Citation

🚀 **If our work sparks inspiration in your research, we’d be thrilled if you could give us a citation!** 📚✨

```

```

And don’t forget to acknowledge the amazing **Jittor framework**: 💻💡

```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}

```

# Acknowledgements
