#!/usr/bin/env python3
"""
Prepare SEED-VIG Extracted Data for Domain Adversarial Training
================================================================

This script converts the preprocessed SEED-VIG EEG data to PSD (Power Spectral
Density) spectral images, matching the format used by TheOriginalEEG pipeline.

PSD Computation:
    The power spectral density for each window is obtained via the squared
    magnitude of the FFT, normalized by the number of samples:
        P(f) = |FFT(x)|^2 / N
    
    The spectrum is log-transformed to compress high-amplitude variability
    and improve detection of subtle spectral changes correlating to drowsiness.
    
    The PSD is normalized per frequency bin, outputting tensors (frequency × 
    channels) representing spectral–spatial structure. This captures 
    physiologically valuable patterns like theta/alpha elevation during drowsiness.

Input:  data/SEED-VIG_Extracted/SEED_VIG.mat
        - EEGsample: (4566, 17, 384) - samples × channels × time points
        - subindex: (4566, 1) - subject IDs (1-12)
        - substate: (4566, 1) - states (0=Alert, 1=Drowsy)

Output: diagnostics/python_data_seedvig/
        - fold_N_data.mat for each subject (LOSO)
        - metadata.mat

The output format matches domain_adversarial_training.py expectations:
        - XTrain/XValidation/XTest: (freq_bins, channels, 1, samples)
        - YTrain_numeric/etc: drowsiness labels (1=Alert, 2=Drowsy)
        - train_subject_nums/etc: subject IDs

Usage:
    python prepare_seedvig_for_dann.py
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.fft import fft
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def compute_psd_features(eeg_data, num_freq_bins=128, fs=128):
    """
    Compute Power Spectral Density (PSD) features from EEG time series.
    
    PSD = |FFT(x)|^2 / N, then log-transformed and normalized per frequency bin.
    
    Args:
        eeg_data: (samples, channels, time_points)
        num_freq_bins: Number of frequency bins to extract
        fs: Sampling frequency in Hz (SEED-VIG is 128 Hz)
        
    Returns:
        psd_features: (freq_bins, channels, 1, samples) - log-transformed PSD
    """
    n_samples, n_channels, n_timepoints = eeg_data.shape
    
    # Compute PSD for each sample and channel
    psd_features = np.zeros((num_freq_bins, n_channels, 1, n_samples))
    
    for i in range(n_samples):
        for ch in range(n_channels):
            # Get time series for this sample and channel
            signal = eeg_data[i, ch, :]
            N = len(signal)
            
            # Apply Hanning window to reduce spectral leakage
            window = np.hanning(N)
            windowed_signal = signal * window
            
            # Compute FFT
            fft_vals = fft(windowed_signal)
            
            # Compute PSD: P(f) = |FFT(x)|^2 / N
            # Take positive frequencies only (up to Nyquist)
            n_fft = N // 2
            psd = (np.abs(fft_vals[:n_fft]) ** 2) / N
            
            # Resample to desired number of frequency bins if needed
            if len(psd) != num_freq_bins:
                indices = np.linspace(0, len(psd) - 1, num_freq_bins)
                psd = np.interp(indices, np.arange(len(psd)), psd)
            
            psd_features[:, ch, 0, i] = psd
    
    # Log-transform to compress high-amplitude variability
    # Add small epsilon to avoid log(0)
    psd_features = np.log10(psd_features + 1e-10)
    
    # Normalize per frequency bin across all samples and channels
    # This standardizes the input for the deep learning model
    for f in range(num_freq_bins):
        freq_slice = psd_features[f, :, :, :]
        mean_val = freq_slice.mean()
        std_val = freq_slice.std() + 1e-8
        psd_features[f, :, :, :] = (freq_slice - mean_val) / std_val
    
    return psd_features


def prepare_loso_folds(eeg_data, labels, subject_ids, output_dir, num_freq_bins=128, fs=128):
    """
    Prepare Leave-One-Subject-Out cross-validation folds.
    
    Args:
        eeg_data: (samples, channels, time_points)
        labels: (samples,) - 0=Alert, 1=Drowsy
        subject_ids: (samples,) - subject IDs (1-12)
        output_dir: Directory to save fold data
        num_freq_bins: Number of PSD frequency bins
        fs: Sampling frequency in Hz
    """
    os.makedirs(output_dir, exist_ok=True)
    
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    n_channels = eeg_data.shape[1]
    
    print(f"\nPreparing {n_subjects} LOSO folds...")
    print(f"Total samples: {len(labels)}")
    print(f"Channels: {n_channels}")
    print(f"PSD frequency bins: {num_freq_bins}")
    print(f"Sampling rate: {fs} Hz")
    
    # Convert all data to PSD features first (more efficient)
    print("\nComputing PSD features for all data...")
    print("  - PSD = |FFT(x)|^2 / N")
    print("  - Log-transformed")
    print("  - Normalized per frequency bin")
    all_spectral = compute_psd_features(eeg_data, num_freq_bins, fs)
    print(f"PSD features shape: {all_spectral.shape}")
    print(f"  Value range: [{all_spectral.min():.2f}, {all_spectral.max():.2f}]")
    print(f"  Mean: {all_spectral.mean():.4f}, Std: {all_spectral.std():.4f}")
    
    # Convert labels: 0=Alert, 1=Drowsy -> 1=Alert, 2=Drowsy (MATLAB convention)
    labels_matlab = labels + 1
    
    fold_info = []
    
    for fold_idx, test_subject in enumerate(unique_subjects, 1):
        print(f"\n--- Fold {fold_idx}: Test Subject {test_subject} ---")
        
        # Split data
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
        
        X_test = all_spectral[:, :, :, test_mask]
        y_test = labels_matlab[test_mask]
        subj_test = subject_ids[test_mask]
        
        X_train_full = all_spectral[:, :, :, train_mask]
        y_train_full = labels_matlab[train_mask]
        subj_train_full = subject_ids[train_mask]
        
        # Split train into train/val (90/10)
        n_train_full = X_train_full.shape[3]
        indices = np.arange(n_train_full)
        
        # Stratified split by label
        train_indices, val_indices = train_test_split(
            indices, test_size=0.1, stratify=y_train_full, random_state=42
        )
        
        X_train = X_train_full[:, :, :, train_indices]
        y_train = y_train_full[train_indices]
        subj_train = subj_train_full[train_indices]
        
        X_val = X_train_full[:, :, :, val_indices]
        y_val = y_train_full[val_indices]
        subj_val = subj_train_full[val_indices]
        
        print(f"  Train: {X_train.shape[3]} samples")
        print(f"  Val:   {X_val.shape[3]} samples")
        print(f"  Test:  {X_test.shape[3]} samples")
        
        # Class balance check
        train_alert = np.sum(y_train == 1)
        train_drowsy = np.sum(y_train == 2)
        test_alert = np.sum(y_test == 1)
        test_drowsy = np.sum(y_test == 2)
        print(f"  Train balance: Alert={train_alert}, Drowsy={train_drowsy}")
        print(f"  Test balance:  Alert={test_alert}, Drowsy={test_drowsy}")
        
        # Save fold data
        fold_data = {
            'XTrain': X_train,
            'YTrain_numeric': y_train.reshape(-1, 1),
            'train_subject_nums': subj_train.reshape(-1, 1),
            
            'XValidation': X_val,
            'YValidation_numeric': y_val.reshape(-1, 1),
            'val_subject_nums': subj_val.reshape(-1, 1),
            
            'XTest': X_test,
            'YTest_numeric': y_test.reshape(-1, 1),
            'test_subject_nums': subj_test.reshape(-1, 1),
            
            'unique_subjects': np.array([f'Subject_{s}' for s in unique_subjects], dtype=object),
            'test_subject_id': test_subject
        }
        
        fold_file = os.path.join(output_dir, f'fold_{fold_idx}_data.mat')
        savemat(fold_file, fold_data, do_compression=True)
        print(f"  Saved: {fold_file}")
        
        fold_info.append({
            'fold': fold_idx,
            'test_subject': int(test_subject),
            'train_samples': X_train.shape[3],
            'val_samples': X_val.shape[3],
            'test_samples': X_test.shape[3]
        })
    
    # Save metadata
    metadata = {
        'num_classes': 2,
        'num_subjects': n_subjects,
        'data_shape': np.array([num_freq_bins, n_channels, 1]),
        'unique_subjects': np.array([f'Subject_{s}' for s in unique_subjects], dtype=object),
        'class_names': np.array(['Alert', 'Drowsy'], dtype=object),
        'dataset': 'SEED-VIG',
        'preprocessing': 'PSD: |FFT|^2/N, log-transformed, normalized per freq bin',
        'sampling_rate': fs
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.mat')
    savemat(metadata_file, metadata)
    print(f"\nSaved metadata: {metadata_file}")
    
    # Save export complete marker
    savemat(os.path.join(output_dir, 'export_complete.mat'), {'complete': True})
    
    # Save fold summary
    with open(os.path.join(output_dir, 'folds.txt'), 'w') as f:
        f.write("SEED-VIG LOSO Folds Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total subjects: {n_subjects}\n")
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"Channels: {n_channels}\n")
        f.write(f"PSD frequency bins: {num_freq_bins}\n")
        f.write(f"Sampling rate: {fs} Hz\n")
        f.write(f"Preprocessing: PSD = |FFT|^2/N, log-transformed, normalized\n\n")
        
        for info in fold_info:
            f.write(f"Fold {info['fold']}: Test Subject {info['test_subject']}\n")
            f.write(f"  Train: {info['train_samples']}, Val: {info['val_samples']}, Test: {info['test_samples']}\n")
    
    print(f"\nSaved fold summary: {os.path.join(output_dir, 'folds.txt')}")
    
    return fold_info


def main():
    print("=" * 60)
    print("SEED-VIG Data Preparation for Domain Adversarial Training")
    print("=" * 60)
    print("\nPSD Computation Method:")
    print("  P(f) = |FFT(x)|^2 / N")
    print("  - Log-transformed to compress high-amplitude variability")
    print("  - Normalized per frequency bin for standardized input")
    
    # Paths
    input_file = 'data/SEED-VIG_Extracted/SEED_VIG.mat'
    output_dir = 'diagnostics/python_data_seedvig'
    
    # Parameters
    num_freq_bins = 128  # Match TheOriginalEEG pipeline
    fs = 128  # SEED-VIG sampling rate is 128 Hz
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    data = loadmat(input_file)
    
    eeg_data = data['EEGsample']  # (4566, 17, 384)
    subject_ids = data['subindex'].flatten()  # (4566,)
    labels = data['substate'].flatten()  # (4566,) - 0=Alert, 1=Drowsy
    
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"  - {eeg_data.shape[0]} samples")
    print(f"  - {eeg_data.shape[1]} channels")
    print(f"  - {eeg_data.shape[2]} time points ({eeg_data.shape[2]/fs:.1f}s @ {fs}Hz)")
    print(f"Subjects: {np.unique(subject_ids)}")
    print(f"Labels: {np.unique(labels)} (0=Alert, 1=Drowsy)")
    
    # Class balance
    n_alert = np.sum(labels == 0)
    n_drowsy = np.sum(labels == 1)
    print(f"Class balance: Alert={n_alert}, Drowsy={n_drowsy}")
    
    # Prepare LOSO folds with PSD features
    fold_info = prepare_loso_folds(
        eeg_data, labels, subject_ids, output_dir, num_freq_bins, fs
    )
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Number of folds: {len(fold_info)}")
    print(f"\nTo train the DANN model, run:")
    print(f"  python domain_adversarial_training.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()

