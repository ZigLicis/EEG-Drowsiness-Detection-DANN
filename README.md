# EEG Drowsiness Detection Pipeline

A hybrid MATLAB-Python deep learning pipeline for subject-independent EEG-based drowsiness detection using Domain Adversarial Neural Networks (DANN).

## 🎯 Overview

This pipeline implements Leave-One-Subject-Out (LOSO) cross-validation with domain adversarial training to achieve robust, subject-independent drowsiness detection from EEG signals. The system processes raw EEG data through frequency-domain transformation and trains CNNs to distinguish between normal (alert) and drowsy states.

**Key Results:** 79.6% ± 15.3% accuracy across 12 subjects using LOSO cross-validation.

## 📁 Pipeline Architecture

```
Raw EEG Data (.cnt files)
    ↓
[MATLAB] Step 1: Preprocessing & Filtering
    ↓
[MATLAB] Step 2: Artifact Removal (Optional)
    ↓
[MATLAB] Step 3: Windowing & FFT Transformation
    ↓
[MATLAB] Subject-wise Normalization & LOSO Split
    ↓
[MATLAB] Export to Python
    ↓
[Python] Domain Adversarial CNN Training
    ↓
[MATLAB] Results Analysis & Visualization
```

## 🔧 Requirements

### MATLAB Dependencies
- EEGLAB (tested with 2025.0.0)
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox (if using MATLAB training)

### Python Dependencies
```bash
pip install -r requirements.txt
```
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21.0
- SciPy ≥ 1.9.0
- Matplotlib ≥ 3.5.0
- scikit-learn ≥ 1.1.0
- h5py ≥ 3.7.0

## 📂 File Structure

### Core Pipeline Files

| File | Description |
|------|-------------|
| `main_dl_pipeline.m` | **Main orchestrator** - Runs entire MATLAB preprocessing pipeline |
| `step1_preprocess_data.m` | Basic EEG preprocessing (filtering, downsampling, re-referencing) |
| `step2_run_ica_and_iclabel.m` | ICA-based artifact removal (optional) |
| `step3_prepare_sequence_data.m` | Windowing and FFT transformation to frequency-domain images |
| `step4_export_for_python.m` | Export processed data for Python deep learning |
| `step4_domain_adversarial_training.py` | **Python DANN training** - Main deep learning script |
| `step5_load_python_results.m` | Load and analyze Python training results |
| `apply_subject_normalization.m` | Subject-wise z-score normalization helper |

### Analysis & Visualization

| File | Description |
|------|-------------|
| `visualize_eeg_samples.py` | Create time-series and frequency-domain visualizations |
| `diagnostics/run_diagnostics.m` | Model diagnostics and batch normalization analysis |

### Configuration

| File | Description |
|------|-------------|
| `requirements.txt` | Python package dependencies |

## 🚀 How to Run the Pipeline

### Step 1: Setup Environment

1. **Install MATLAB dependencies:**
   ```matlab
   % In MATLAB, add EEGLAB to path
   addpath('/path/to/eeglab2025.0.0')
   eeglab
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Prepare Data

Organize your EEG data in the following structure:
```
/path/to/your/data/
├── Subject1/
│   ├── Normal state.cnt
│   └── Fatigue state.cnt
├── Subject2/
│   ├── Normal state.cnt
│   └── Fatigue state.cnt
└── ...
```

### Step 3: Configure Pipeline

Edit `main_dl_pipeline.m` to set your data path:
```matlab
data_root_path = '/path/to/your/EEG/data';
```

### Step 4: Run MATLAB Preprocessing

```matlab
% In MATLAB command window
main_dl_pipeline
```

This will:
- Process all subjects' EEG data
- Apply filtering, windowing, and FFT transformation
- Perform subject-wise normalization
- Create LOSO cross-validation splits
- Export data to `python_data/` directory

### Step 5: Run Python Deep Learning

```bash
python step4_domain_adversarial_training.py
```

This will:
- Load exported MATLAB data
- Train Domain Adversarial CNNs for each LOSO fold
- Save trained models and results
- Generate confusion matrices

### Step 6: Analyze Results

```matlab
% In MATLAB
step5_load_python_results
```

This will:
- Load Python training results
- Display accuracy statistics and variance analysis
- Generate summary plots and recommendations

## 📊 Pipeline Details

### Preprocessing (MATLAB)

**Step 1: Basic Preprocessing**
- **Channels:** Frontal EEG channels (FP1, FP2, F7, F3, FZ, F4, F8)
- **Filtering:** 0.5-50 Hz bandpass filter
- **Sampling:** Downsampled to 250 Hz
- **Reference:** Linked mastoids (A1, A2)

**Step 2: Artifact Removal (Optional)**
- **ICA:** Independent Component Analysis
- **ICLabel:** Automatic artifact component classification
- **Note:** Currently disabled to test impact on performance

**Step 3: Data Preparation**
- **Windowing:** 5-second non-overlapping windows
- **FFT:** 1024-point FFT → 128 frequency bins (0-31.25 Hz)
- **Format:** Frequency × Channels × 1 images for CNN input

**Step 4: Subject-wise Normalization**
- **Method:** Z-score normalization per subject
- **Training:** Computed from training subjects only
- **Test:** Each test subject normalized by their own statistics

### Deep Learning (Python)

**Domain Adversarial Neural Network (DANN)**
- **Architecture:** 3-block 2D CNN with shared feature extractor
- **Branches:** 
  - Drowsiness classifier (main task)
  - Subject classifier (adversarial task)
- **Gradient Reversal:** Forces subject-invariant feature learning
- **Training:** Adam optimizer, early stopping, L2 regularization

**Cross-Validation**
- **Method:** Leave-One-Subject-Out (LOSO)
- **Folds:** 12 (one per subject)
- **Split:** Train/Validation/Test = 80%/20%/100% (of remaining subjects)

## 📈 Expected Results

### Performance Metrics
- **Mean Accuracy:** ~79.6%
- **Standard Deviation:** ~15.3%
- **Range:** 50-100% across subjects
- **Best Folds:** Often achieve >95% accuracy
- **Challenging Subjects:** Some subjects remain difficult (~50-70%)

### Frequency Band Analysis
- **Delta (0.5-4 Hz):** ↑ Increased in drowsiness ✅
- **Theta (4-8 Hz):** ↑ Increased in drowsiness ✅  
- **Alpha (8-13 Hz):** ↑ Increased in drowsiness ✅
- **Beta (13-30 Hz):** Variable (may contain artifacts)

## 🔧 Customization Options

### Modify Preprocessing Parameters
In `main_dl_pipeline.m`:
```matlab
low_cutoff_freq  = 0.5;    % Low-pass filter
high_cutoff_freq = 50;     % High-pass filter
downsample_rate  = 250;    % Sampling rate
window_length_sec = 5;     % Window size
```

### Adjust Deep Learning Parameters
In `step4_domain_adversarial_training.py`:
```python
num_epochs = 80           # Training epochs
lr = 0.001               # Learning rate
batch_size = 32          # Mini-batch size
lambda_ = 0.5            # Adversarial strength
```

### Change Cross-Validation Strategy
- **K-Fold:** Modify `k_folds` in `main_dl_pipeline.m`
- **Random Split:** Replace LOSO logic with random subject assignment
- **Stratified:** Implement balanced class splitting

## 🐛 Troubleshooting

### Common Issues

**1. MATLAB: "pop_loadcnt function not found"**
```matlab
% Solution: Ensure EEGLAB is properly initialized
eeglab nogui
```

**2. Python: "No such file or directory: metadata.mat"**
```bash
# Solution: Run MATLAB preprocessing first
# Check that python_data/ directory exists with exported files
```

**3. High Cross-Subject Variance**
- Check data quality and preprocessing consistency
- Consider additional artifact removal
- Adjust domain adversarial training strength
- Implement data augmentation

**4. Low Overall Accuracy**
- Verify frequency band analysis shows expected patterns
- Check for label consistency across subjects
- Consider different CNN architectures
- Validate preprocessing parameters

### Debug Mode

Enable verbose output:
```matlab
% In main_dl_pipeline.m, add:
fprintf('Debug: Processing subject %s\n', subject_id);
```

```python
# In step4_domain_adversarial_training.py, set:
'Verbose': True  # in trainingOptions
```

## 📚 References

### Key Papers
- Domain Adversarial Training: Ganin et al. (2016)
- EEG Drowsiness Detection: Chaabene et al. (2021)
- Subject-Independent EEG: Various LOSO studies

### EEG Frequency Bands
- **Delta (0.5-4 Hz):** Deep sleep, unconsciousness
- **Theta (4-8 Hz):** Drowsiness, light sleep, meditation
- **Alpha (8-13 Hz):** Relaxed wakefulness, eyes closed
- **Beta (13-30 Hz):** Alert, focused attention, active thinking

## 🤝 Contributing

To extend this pipeline:
1. **Add new preprocessing steps:** Modify `step1_preprocess_data.m`
2. **Implement new architectures:** Create variants of the Python training script
3. **Add evaluation metrics:** Extend `step5_load_python_results.m`
4. **Optimize hyperparameters:** Use grid search or Bayesian optimization

## 📄 License

This project is provided for research and educational purposes. Please cite appropriately if used in publications.

---

**Pipeline developed for EEG-based drowsiness detection research**  
*Hybrid MATLAB-Python implementation with Domain Adversarial Training* 