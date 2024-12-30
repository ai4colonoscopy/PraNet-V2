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
# We use Python3.9, CUDA 12.2, PyTorch2.0.1.
conda env create -f pranet2.yaml
```

#### Dataset Preparation

1. Polyp datasets
   - To download the testing dataset, use this [Google Drive Link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing) and place it in the directory `./data/TestDataset/`.
   - For the training dataset, download it from this [Google Drive Link (399.5MB)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing) and move it to `./data/TrainDataset/`.
2. **ACDC and Synapse dataset**
   - Please refer to the [EMCAD](https://github.com/SLDGroup/EMCAD) for the preprocessed dataset used in this project. Make sure to follow their guidelines for proper usage.

#### Weights Preparation

💾 **Need PVTV2B2 and Res2Net weights**， Please [click here to download](https://drive.google.com/drive/folders/17VuWBy6CnEXF7kASydQGsZgYFjiwhez2?usp=sharing) 🎯

# Interface

Before running the inference script, make sure to grab the **pre-trained model weights** 🔗 [Download here](https://drive.google.com/drive/folders/1o4UHzI48wwEtpz-J91dnIRhtYy67lFCO?usp=sharing).



# Evaluation





# Training

### PraNet-V2



### PVT-PraNet-V2



### MERIT



### MIST



### EMCAD





# Bibtex 

