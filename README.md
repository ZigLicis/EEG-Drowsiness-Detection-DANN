# EEG Drowsiness Detection with Domain Adversarial Neural Networks

Cross-subject EEG-based drowsiness/fatigue detection using Domain Adversarial Neural Networks (DANN) with gradient reversal to learn subject-invariant features.

## Overview

This project implements a complete pipeline for:
1. **EEG Preprocessing** (MATLAB/EEGLAB): Filtering, ASR, ICA artifact removal
2. **Feature Extraction**: FFT spectral images from 5-second windows
3. **Deep Learning** (PyTorch): Domain Adversarial CNN with gradient reversal

The DANN architecture learns features that are:
- **Discriminative** for drowsiness detection (main task)
- **Invariant** to subject identity (adversarial task via gradient reversal)

## Supported Datasets

- **TheOriginalEEG**: 12 subjects, Normal vs Fatigued states
- **SADT**: 11 subjects, preprocessed spectral features
- **SEED-VIG**: 12 subjects, vigilance estimation during driving

## Project Structure

```
├── main_dl_pipeline.m          # Main MATLAB preprocessing pipeline
├── step1_preprocess_data.m     # Filtering, downsampling, ASR
├── step2_run_ica_and_iclabel.m # ICA + ICLabel artifact removal
├── step3_prepare_sequence_data.m # FFT windowing
├── step4_export_for_python.m   # Export to .mat for Python
│
├── domain_adversarial_training.py  # Main DANN training script
├── train_dann_originaleeg.py       # DANN for TheOriginalEEG
├── train_dann_sadt.py              # DANN for SADT dataset
├── train_dann_seedvig.py           # DANN for SEED-VIG dataset
├── train_dann_combined.py          # DANN for combined datasets
│
├── analyze_subject_characteristics.m  # Subject analysis utilities
└── visualize_eeg_timeseries.m         # Visualization tools
```

## Requirements

### MATLAB (Preprocessing)
- MATLAB R2020a or later
- EEGLAB (with plugins: clean_rawdata, ICLabel, firfilt)

### Python (Training)
```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn h5py
```

## Quick Start

### 1. Preprocessing (MATLAB)
```matlab
% Edit data paths in main_dl_pipeline.m, then run:
main_dl_pipeline
```

### 2. Training (Python)
```bash
# For TheOriginalEEG dataset
python train_dann_originaleeg.py

# For SADT dataset
python train_dann_sadt.py

# For combined datasets
python domain_adversarial_training.py --data_dir diagnostics/python_data
```

## Google Colab Setup

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/drowsiness-detection.git
%cd drowsiness-detection

# Install dependencies
!pip install torch torchvision scipy matplotlib scikit-learn h5py

# Upload your preprocessed data to Google Drive and mount
from google.colab import drive
drive.mount('/content/drive')

# Copy data from Drive (adjust path as needed)
!cp -r /content/drive/MyDrive/drowsiness_data/python_data diagnostics/

# Run training with GPU
!python domain_adversarial_training.py --data_dir diagnostics/python_data
```

## Data Setup

The raw EEG data files are not included in this repository due to size constraints.

### Option 1: Preprocess locally, upload features
1. Run MATLAB preprocessing locally
2. Upload `diagnostics/python_data/*.mat` to Google Drive
3. Mount Drive in Colab and copy files

### Option 2: Download datasets directly
- **TheOriginalEEG**: [Figshare Link]
- **SADT**: [Dataset Source]
- **SEED-VIG**: [BCMI Lab](http://bcmi.sjtu.edu.cn/~seed/seed-vig.html)

## Model Architecture

```
Input: FFT Spectral Image (128 freq × 7 channels)
         │
    ┌────▼────┐
    │  CNN    │  Shared Feature Extractor
    │ Encoder │  (4 conv blocks, BatchNorm, ELU)
    └────┬────┘
         │
    ┌────▼────┐
    │ FC 256  │  Shared Features
    └────┬────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│Drowsy │ │  GRL  │  Gradient Reversal Layer
│ Head  │ └───┬───┘
└───┬───┘     │
    │    ┌────▼────┐
    │    │Subject  │
    │    │  Head   │
    │    └────┬────┘
    ▼         ▼
 Normal/   Subject ID
 Fatigued  (adversarial)
```

## Results

| Dataset | Balanced Accuracy | Subject Invariance |
|---------|-------------------|-------------------|
| TheOriginalEEG | ~70% | Near chance |
| SADT | ~75% | Near chance |
| SEED-VIG | ~72% | Near chance |

## Citation

If you use this code, please cite:
```
@misc{drowsiness-dann,
  author = {Your Name},
  title = {EEG Drowsiness Detection with Domain Adversarial Networks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/drowsiness-detection}
}
```

## License

MIT License - see LICENSE file for details.

