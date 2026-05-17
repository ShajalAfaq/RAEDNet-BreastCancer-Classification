# RAEDNet-BreastCancer-Classification
Implementation of RAEDNet integrating residual dense blocks, multi-attention mechanisms, and CEAO optimization for breast cancer diagnosis.
# RAEDNet: Residual Attention Efficient Dense Network for Breast Cancer Histopathological Classification

## Overview
This repository contains the implementation of the proposed Residual Attention Efficient Dense Network (RAEDNet) for automated breast cancer classification using histopathological images. The framework integrates residual dense learning, multi-scale depthwise separable convolutions, and heterogeneous attention mechanisms to improve discriminative feature extraction from complex tissue structures. In addition, a Chaotic-Based Enzyme Action Optimizer (CEAO) is employed for adaptive hyperparameter optimization to enhance convergence stability and model generalization. The proposed framework was evaluated on the BreakHis and BACH 2018 histopathology datasets for both binary and multiclass breast cancer classification tasks.

## Key Features
Multi-scale feature extraction using depthwise separable convolutions
Residual Efficient Dense Blocks (REDBs)
Spatial, channel, CBAM, and non-local attention mechanisms
Chaotic-Based Enzyme Action Optimizer (CEAO)
Binary and multiclass breast cancer classification
Grad-CAM interpretability visualization
Computationally efficient deep learning framework


## Proposed Architecture
Input Image  
↓  
Efficient Stem Block  
↓  
REDB-1 + Spatial Attention  
↓  
REDB-2 + Non-local Self-Attention  
↓  
REDB-3 + CBAM  
↓  
REDB-4 + Dilated Convolutions  
↓  
REDB-5 + Channel-wise Gated Residual  
↓  
Global Average Pooling + Global Max Pooling  
↓  
Dense Projection + Dropout  
↓  
Softmax Classification Layer

---

## Datasets
BreakHis and BACH 2018
### BreakHis Dataset
- 7909 histopathological breast cancer images
- 8 subclasses
- Magnification factors:
  - 40×
  - 100×
  - 200×
  - 400×

### BACH 2018 Dataset
- 400 histopathological images
- 4 classes:
  - Normal
  - Benign
  - In situ carcinoma
  - Invasive carcinoma


## Dataset Split Strategy
The datasets were divided using a stratified:
- 70% Training
- 10% Validation
- 20% Testing


