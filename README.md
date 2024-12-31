<h1 align="center">PraNet-V2: Dual-Supervised Reverse Attention for Medical Image Segmentation</h1>

<div align='center'>
    <strong>Bo-Cheng Hu</strong></a><sup>1</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=oaxKYKUAAAAJ' target='_blank'><strong>Ge-Peng Ji</strong></a><sup> 2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=amxDSLoAAAAJ' target='_blank'><strong>Dian Shao</strong></a><sup> 3</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=kakwJ5QAAAAJ' target='_blank'><strong>Deng-Ping Fan</strong></a><sup> 1</sup>,&thinsp;
</div>

<div align='center'>
    <sup>1 </sup>Nankai University&ensp;  <sup>2 </sup>Australian National University&ensp;  <sup>3 </sup>Northwestern Polytechnical University&ensp; 
</div>





# 🔥News🔥



# usage

#### Environment Setup

```bash
git clone git@github.com:ai4colonoscopy/PraNet-V2.git
cd PraNet-V2
# We use Python3.9, CUDA 12.2, PyTorch2.0.1.
conda env create -f pranet2.yaml
conda activate pranet2
```

#### Dataset Preparation

1. **Polyp datasets**
   - To download the testing dataset, use this [Google Drive Link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing) and place it in the directory `./data/TestDataset/`.
   - For the training dataset, download it from this [Google Drive Link (399.5MB)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing) and move it to `./data/TrainDataset/`.
2. **ACDC and Synapse datasets**
   - Please refer to the [EMCAD](https://github.com/SLDGroup/EMCAD) for the preprocessed dataset used in this project. Make sure to follow their guidelines for proper usage.

#### Weights Preparation

💾 **Need PVTV2B2 and Res2Net weights**， Please [click here to download](xxx) 🎯

# Interface

📦 Before running the inference script, make sure to grab the **pre-trained model weights** 🔗 [Download here](xxx). Then, organize the PraNet series models under `./binary/snapshots` following the structure below:

```
snapshots
├── PraNet-V1
│   └── RES-V1.pth
├── PraNet-V2
│   └── RES-V2.pth
├── PVT-PraNet-V2
│   └── PVT-V2.pth
└── PVT-V1
    └── PVT-V1.pth
```

For the multi-class segmentation models, place them in their respective model_pth folders. For example:

`./multi-class/EMCAD/model_pth/ACDC/EMCAD_ACDC.pth`

Then, run the corresponding inference scripts to generate segmentation or test results:

1. PraNet-V2 series： Please make sure to fill in the model paths and segmentation result save paths as guided by the TODO markers and run `python -W ignore ./binary/MyTest_med.py`
2. Multi-class segmentation models:

```bash
# MIST
python -W ignore Synapse_test.py 
python -W ignore test_ACDC.py

# EMCAD
python -W ignore test_synapse.py
```





# Evaluation

For the **PraNet series models**, follow the **Interface** steps to generate segmentation results, which will be saved in the results folder. Afterward, run the `eval.py` script to generate a performance evaluation table in the `eval_results` folder.

```
python -W ignore ./binary/eval.py
```



The `eval.py` script provides **four eval_config options** for evaluating the performance of the following models: PraNet-V1、PVT-PraNet-V1、PraNet-V2、PVT-PraNet-V2. You can try different configs in the script to check out the evaluation results for these models.

For the three **multi-class segmentation models**, the evaluation results are already logged during the **Interface** step

# Training

### PraNet-V2



### PVT-PraNet-V2



### MERIT



### MIST



### EMCAD





# Bibtex 





# Training

### PraNet-V2



### PVT-PraNet-V2



### MERIT



### MIST



### EMCAD





# Bibtex 

