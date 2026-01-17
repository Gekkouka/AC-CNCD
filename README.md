# AC-CNCD
This repository contains the official implementation of the paper “AC-CDCN:A Cross-Subject EEG Emotion Recognition Model with Anti-Collapse Domain Generalization”. 

## Overview
AC-CDCN addresses cross-subject EEG emotion recognition under the domain generalization setting, 
where each subject is treated as an independent domain and no target-subject data is accessible during training.

## Key Features
- Cross-subject EEG emotion recognition under domain generalization
- Anti-collapse regularization for robust representation learning
- Subject-invariant EEG feature learning without target-domain access
- Reproducible experimental pipeline

## Datasets
This code supports experiments on the following public EEG emotion datasets:
- SEED
- SEED-IV
Please follow the original dataset licenses and download instructions.

## Usage
```bash
python main_seed.py
```

## License
This project is licensed under the MIT License.
