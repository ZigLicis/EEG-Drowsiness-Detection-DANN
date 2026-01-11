#!/usr/bin/env python3
"""
Prepare SEED-VIG Extracted Data for DANN Pipeline
==================================================

This script converts the pre-extracted SEED-VIG data (3-second labeled windows)
to PSD format matching the TheOriginalEEG pipeline, and exports LOSO folds.

SEED-VIG data structure:
- EEGsample: (4566, 17, 384) - 4566 samples, 17 channels, 384 timepoints (3s @ 128Hz)
- subindex: (4566, 1) - Subject index (1-12)
- substate: (4566, 1) - Drowsiness state (0=alert, 1=drowsy)

Output format matches step4_export_for_python.m:
- fold_X_data.mat with XTrain, YTrain_numeric, train_subject_nums, etc.
- metadata.mat with num_classes, num_subjects, data_shape, unique_subjects
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

def compute_psd(eeg_segment, sfreq=128, num_freq_bins=128):
    """
    Compute Power Spectral Density for an EEG segment.
    
    Matches the PSD calculation in step3_prepare_sequence_data.m:
    - FFT of the signal
    - Squared magnitude normalized by N
    - Log transform
    
    Args:
        eeg_segment: (channels, timepoints) array
        sfreq: Sampling frequency (SEED-VIG is 128 Hz based on 384 samples / 3s)
        num_freq_bins: Number of frequency bins to keep
        
    Returns:
        psd: (freq_bins, channels, 1) array matching MATLAB output format
    """
    n_channels, n_samples = eeg_segment.shape
    
    # Compute FFT - use power of 2 for efficiency
    nfft = max(256, 2 ** int(np.ceil(np.log2(n_samples))))
    spectrum = fft(eeg_segment, n=nfft, axis=1)
    
    # Compute PSD: |FFT|^2 / N
    psd = np.abs(spectrum) ** 2 / n_samples
    
    # Keep only positive frequencies up to num_freq_bins
    # For 128 Hz sampling, Nyquist is 64 Hz
    # With nfft=512, freq resolution is 128/512 = 0.25 Hz
    psd = psd[:, :num_freq_bins]
    
    # Log transform (with small epsilon for numerical stability)
    psd = np.log10(psd + 1e-10)
    
    # Rearrange to (freq_bins, channels, 1) to match MATLAB format
    psd = psd.T  # (freq_bins, channels)
    psd = psd[:, :, np.newaxis]  # (freq_bins, channels, 1)
    
    return psd

def prepare_seedvig_for_dann(input_path, output_dir, num_freq_bins=128):
    """
    Convert SEED-VIG data to PSD format and create LOSO folds.
    
    Args:
        input_path: Path to SEED_VIG.mat
        output_dir: Directory to save fold files
        num_freq_bins: Number of frequency bins for PSD
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading SEED-VIG data from: {input_path}")
    data = loadmat(input_path)
    
    # Extract arrays
    # EEGsample: (n_samples, n_channels, n_timepoints)
    # subindex: (n_samples, 1) - subject index (1-12)
    # substate: (n_samples, 1) - drowsiness state (0=alert, 1=drowsy)
    
    eeg_samples = data['EEGsample']  # (4566, 17, 384)
    subject_indices = data['subindex'].flatten()  # (4566,)
    drowsiness_labels = data['substate'].flatten()  # (4566,)
    
    n_samples, n_channels, n_timepoints = eeg_samples.shape
    
    print(f"\nData loaded:")
    print(f"  EEG samples: {eeg_samples.shape}")
    print(f"  Subjects: {np.unique(subject_indices)} (n={len(np.unique(subject_indices))})")
    print(f"  Labels: {np.unique(drowsiness_labels)} (0=alert, 1=drowsy)")
    print(f"  Samples per class: alert={np.sum(drowsiness_labels==0)}, drowsy={np.sum(drowsiness_labels==1)}")
    
    # Compute sampling frequency
    # 384 timepoints / 3 seconds = 128 Hz
    sfreq = n_timepoints / 3.0
    print(f"  Sampling frequency: {sfreq} Hz")
    
    # Convert all samples to PSD
    print(f"\nConverting {n_samples} samples to PSD format...")
    all_psd = []
    
    for i in range(n_samples):
        # Get segment: (channels, timepoints)
        segment = eeg_samples[i, :, :]  # (17, 384)
        
        # Compute PSD
        psd = compute_psd(segment, sfreq=int(sfreq), num_freq_bins=num_freq_bins)
        all_psd.append(psd)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{n_samples} samples")
    
    # Stack into array: (freq_bins, channels, 1, n_samples)
    X_all = np.stack(all_psd, axis=3)
    
    # Labels: convert to 1-based (1=alert, 2=drowsy) to match MATLAB convention
    y_all = drowsiness_labels + 1
    
    # Subject indices are already 1-based in SEED-VIG
    subjects_all = subject_indices
    
    print(f"\nFinal data:")
    print(f"  X shape: {X_all.shape}")
    print(f"  y shape: {y_all.shape}, unique: {np.unique(y_all)}")
    print(f"  subjects shape: {subjects_all.shape}, unique: {np.unique(subjects_all)}")
    
    # Create LOSO folds
    create_loso_folds(X_all, y_all, subjects_all, output_dir)

def create_loso_folds(X_all, y_all, subjects_all, output_dir):
    """
    Create Leave-One-Subject-Out folds matching the MATLAB export format.
    
    Args:
        X_all: (freq_bins, channels, 1, n_samples) array
        y_all: (n_samples,) array of labels (1-based)
        subjects_all: (n_samples,) array of subject indices (1-based)
        output_dir: Directory to save fold files
    """
    
    unique_subjects = np.unique(subjects_all)
    n_subjects = len(unique_subjects)
    
    print(f"\n{'='*60}")
    print(f"Creating {n_subjects} LOSO folds...")
    print(f"{'='*60}")
    
    for fold_idx, test_subject in enumerate(unique_subjects):
        fold_num = fold_idx + 1
        print(f"\n--- Fold {fold_num}/{n_subjects}: Test subject = {test_subject} ---")
        
        # Split data
        test_mask = subjects_all == test_subject
        train_val_mask = ~test_mask
        
        # Get indices
        train_val_indices = np.where(train_val_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Split train/val (80/20)
        np.random.seed(42 + fold_idx)  # Reproducible split
        n_train_val = len(train_val_indices)
        n_val = int(0.2 * n_train_val)
        
        shuffled_indices = np.random.permutation(train_val_indices)
        val_indices = shuffled_indices[:n_val]
        train_indices = shuffled_indices[n_val:]
        
        # Extract data
        XTrain = X_all[:, :, :, train_indices]
        XValidation = X_all[:, :, :, val_indices]
        XTest = X_all[:, :, :, test_indices]
        
        YTrain_numeric = y_all[train_indices].astype(np.float64)
        YValidation_numeric = y_all[val_indices].astype(np.float64)
        YTest_numeric = y_all[test_indices].astype(np.float64)
        
        train_subject_nums = subjects_all[train_indices].astype(np.float64)
        val_subject_nums = subjects_all[val_indices].astype(np.float64)
        test_subject_nums = subjects_all[test_indices].astype(np.float64)
        
        # Print fold statistics
        train_labels = YTrain_numeric.astype(int)
        test_labels = YTest_numeric.astype(int)
        print(f"  Train: {XTrain.shape[3]} samples (alert={np.sum(train_labels==1)}, drowsy={np.sum(train_labels==2)})")
        print(f"  Val: {XValidation.shape[3]} samples")
        print(f"  Test: {XTest.shape[3]} samples (alert={np.sum(test_labels==1)}, drowsy={np.sum(test_labels==2)})")
        
        # Save fold data
        fold_file = os.path.join(output_dir, f'fold_{fold_num}_data.mat')
        
        # Create unique_subjects as cell array of strings
        unique_subjects_list = [f'Subject_{i}' for i in range(1, n_subjects + 1)]
        
        save_data = {
            'XTrain': XTrain,
            'YTrain_numeric': YTrain_numeric.reshape(-1, 1),
            'train_subject_nums': train_subject_nums.reshape(-1, 1),
            'XValidation': XValidation,
            'YValidation_numeric': YValidation_numeric.reshape(-1, 1),
            'val_subject_nums': val_subject_nums.reshape(-1, 1),
            'XTest': XTest,
            'YTest_numeric': YTest_numeric.reshape(-1, 1),
            'test_subject_nums': test_subject_nums.reshape(-1, 1),
            'unique_subjects': np.array(unique_subjects_list, dtype=object),
            'fold_num': fold_num
        }
        
        savemat(fold_file, save_data, do_compression=True)
        print(f"  Saved: {fold_file}")
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.mat')
    metadata = {
        'num_classes': int(len(np.unique(y_all))),
        'num_subjects': int(n_subjects),
        'data_shape': np.array(X_all.shape),
        'unique_subjects': np.array([f'Subject_{i}' for i in range(1, n_subjects + 1)], dtype=object),
        'sfreq': 128,
        'window_length_sec': 3,
        'num_channels': X_all.shape[1],
        'num_freq_bins': X_all.shape[0]
    }
    savemat(metadata_file, metadata)
    print(f"\nMetadata saved: {metadata_file}")
    
    # Save export complete flag
    export_file = os.path.join(output_dir, 'export_complete.mat')
    savemat(export_file, {'k_folds': n_subjects, 'export_complete': True})
    print(f"Export complete flag saved: {export_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {X_all.shape[3]}")
    print(f"Data shape per sample: ({X_all.shape[0]} freq bins, {X_all.shape[1]} channels, 1)")
    print(f"Number of subjects: {n_subjects}")
    print(f"Number of folds: {n_subjects}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare SEED-VIG data for DANN pipeline')
    parser.add_argument('--input', type=str, 
                        default='data/VLA_VRW_extracted.mat',
                        help='Path to SEED_VIG.mat file')
    parser.add_argument('--output', type=str,
                        default='diagnostics/python_data_seedvla_psd_',
                        help='Output directory for fold files')
    parser.add_argument('--freq_bins', type=int, default=128,
                        help='Number of frequency bins for PSD')
    
    args = parser.parse_args()
    
    prepare_seedvig_for_dann(args.input, args.output, args.freq_bins)
    
    print("\n" + "="*60)
    print("SEED-VIG data preparation complete!")
    print("="*60)
    print(f"\nTo run ablation study on this data:")
    print(f"  python ablation_study.py --data_dir {args.output}")
