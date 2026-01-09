#!/usr/bin/env python3
"""
Prepare SEED-VLA and SEED-VRW Datasets for Domain Adversarial Training
=======================================================================

This script preprocesses the raw SEED-VLA (lab) and SEED-VRW (real-world) EEG 
datasets following the methodology used for SEED-VIG:

Preprocessing Pipeline:
    1. Load raw EEG from .edf files (300 Hz, 24 channels)
    2. Downsample to 128 Hz
    3. Apply 1 Hz high-pass filter (remove DC drift)
    4. Extract 3-second segments prior to each PERCLOS evaluation
    5. Label segments: Alert (PERCLOS < threshold) / Drowsy (PERCLOS > threshold)
    6. Discard middle-range samples
    7. Balance classes per subject
    8. Compute PSD features: P(f) = |FFT(x)|^2 / N, log-transformed
    9. Export LOSO folds for domain_adversarial_training.py

Dataset Info:
    - VLA (Lab): 20 subjects, ~28 hours, simulated driving in lab
    - VRW (Real): 14 subjects, ~21 hours, real-world driving
    - Channels: 18 standard 10-20 electrodes (referenced to Pz)
    - PERCLOS: Computed from eye-tracking, sampled every ~9 seconds

Reference:
    Luo et al. "A cross-scenario and cross-subject domain adaptation method 
    for driving fatigue detection" J. Neural Eng. 2024

Usage:
    python prepare_vla_vrw_for_dann.py [--alert_thresh 0.35] [--drowsy_thresh 0.70]
"""

import os
import numpy as np
import mne
from scipy.io import loadmat, savemat
from scipy.fft import fft
from scipy.signal import resample, butter, filtfilt
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')

# Standard 10-20 channel names (excluding reference/auxiliary channels)
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
    'C3', 'C4', 'Cz', 'T3', 'T4', 'T5', 'T6',
    'P3', 'P4', 'O1', 'O2'
]

# Channel name mapping from EDF format to standard names
CHANNEL_MAP = {
    'EEG Fp1 - Pz': 'Fp1', 'EEG Fp2 - Pz': 'Fp2',
    'EEG F3 - Pz': 'F3', 'EEG F4 - Pz': 'F4',
    'EEG F7 - Pz': 'F7', 'EEG F8 - Pz': 'F8',
    'EEG Fz - Pz': 'Fz',
    'EEG C3 - Pz': 'C3', 'EEG C4 - Pz': 'C4',
    'EEG Cz - Pz': 'Cz',
    'EEG T3 - Pz': 'T3', 'EEG T4 - Pz': 'T4',
    'EEG T5 - Pz': 'T5', 'EEG T6 - Pz': 'T6',
    'EEG P3 - Pz': 'P3', 'EEG P4 - Pz': 'P4',
    'EEG O1 - Pz': 'O1', 'EEG O2 - Pz': 'O2',
}


def highpass_filter(data, cutoff=1.0, fs=300, order=4):
    """Apply high-pass Butterworth filter."""
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype='high')
    return filtfilt(b, a, data, axis=1)


def extract_segments(eeg_data, perclos_values, fs_original=300, fs_target=128,
                     segment_duration=3.0, alert_thresh=0.35, drowsy_thresh=0.70):
    """
    Extract labeled EEG segments aligned with PERCLOS evaluations.
    
    Args:
        eeg_data: (channels, samples) raw EEG at fs_original
        perclos_values: (n_perclos,) PERCLOS values
        fs_original: Original sampling rate (300 Hz)
        fs_target: Target sampling rate (128 Hz)
        segment_duration: Duration of each segment in seconds
        alert_thresh: PERCLOS threshold for alert state
        drowsy_thresh: PERCLOS threshold for drowsy state
        
    Returns:
        segments: List of (channels, time) arrays
        labels: List of labels (0=alert, 1=drowsy)
    """
    n_channels, n_samples = eeg_data.shape
    n_perclos = len(perclos_values)
    
    # Calculate PERCLOS interval (time between evaluations)
    total_duration = n_samples / fs_original
    perclos_interval = total_duration / n_perclos
    
    # Segment parameters
    segment_samples_orig = int(segment_duration * fs_original)
    segment_samples_target = int(segment_duration * fs_target)
    
    # Skip first 24 seconds as recommended in README
    skip_samples = int(24 * fs_original)
    skip_perclos = int(24 / perclos_interval)
    
    segments = []
    labels = []
    
    for i in range(skip_perclos, n_perclos):
        perclos = perclos_values[i]
        
        # Determine label
        if perclos < alert_thresh:
            label = 0  # Alert
        elif perclos > drowsy_thresh:
            label = 1  # Drowsy
        else:
            continue  # Skip middle range
        
        # Calculate segment end time (at PERCLOS evaluation)
        segment_end_time = (i + 1) * perclos_interval
        segment_end_sample = int(segment_end_time * fs_original)
        segment_start_sample = segment_end_sample - segment_samples_orig
        
        # Check bounds
        if segment_start_sample < skip_samples:
            continue
        if segment_end_sample > n_samples:
            continue
        
        # Extract segment
        segment = eeg_data[:, segment_start_sample:segment_end_sample]
        
        # Resample to target frequency
        segment_resampled = resample(segment, segment_samples_target, axis=1)
        
        segments.append(segment_resampled)
        labels.append(label)
    
    return segments, labels


def compute_psd_features(segments, fs=128, num_freq_bins=128):
    """
    Compute Power Spectral Density features from EEG segments.
    
    PSD = |FFT(x)|^2 / N, then log-transformed and normalized per frequency bin.
    
    Args:
        segments: List of (channels, time) arrays
        fs: Sampling frequency
        num_freq_bins: Number of frequency bins to extract
        
    Returns:
        psd_features: (freq_bins, channels, 1, samples)
    """
    n_samples = len(segments)
    n_channels = segments[0].shape[0]
    
    psd_features = np.zeros((num_freq_bins, n_channels, 1, n_samples))
    
    for i, segment in enumerate(segments):
        for ch in range(n_channels):
            signal = segment[ch, :]
            N = len(signal)
            
            # Apply Hanning window
            window = np.hanning(N)
            windowed_signal = signal * window
            
            # Compute PSD: P(f) = |FFT(x)|^2 / N
            fft_vals = fft(windowed_signal)
            n_fft = N // 2
            psd = (np.abs(fft_vals[:n_fft]) ** 2) / N
            
            # Resample to desired frequency bins
            if len(psd) != num_freq_bins:
                indices = np.linspace(0, len(psd) - 1, num_freq_bins)
                psd = np.interp(indices, np.arange(len(psd)), psd)
            
            psd_features[:, ch, 0, i] = psd
    
    # Log-transform
    psd_features = np.log10(psd_features + 1e-10)
    
    # Normalize per frequency bin
    for f in range(num_freq_bins):
        freq_slice = psd_features[f, :, :, :]
        mean_val = freq_slice.mean()
        std_val = freq_slice.std() + 1e-8
        psd_features[f, :, :, :] = (freq_slice - mean_val) / std_val
    
    return psd_features


def process_subject(eeg_path, perclos_path, alert_thresh, drowsy_thresh, min_samples=50):
    """
    Process a single subject's data.
    
    Returns:
        segments: List of preprocessed EEG segments
        labels: List of labels
        or None if subject doesn't meet criteria
    """
    # Load EEG
    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
    
    # Get channel indices for standard channels
    ch_indices = []
    ch_names_ordered = []
    for ch_name in raw.ch_names:
        if ch_name in CHANNEL_MAP:
            ch_indices.append(raw.ch_names.index(ch_name))
            ch_names_ordered.append(CHANNEL_MAP[ch_name])
    
    # Extract data for selected channels
    data = raw.get_data()[ch_indices, :]  # (channels, samples)
    fs = raw.info['sfreq']
    
    # Apply 1 Hz high-pass filter
    data = highpass_filter(data, cutoff=1.0, fs=fs)
    
    # Load PERCLOS
    perclos_data = loadmat(perclos_path)['perclos'].flatten()
    
    # Extract segments
    segments, labels = extract_segments(
        data, perclos_data, 
        fs_original=fs, fs_target=128,
        alert_thresh=alert_thresh, drowsy_thresh=drowsy_thresh
    )
    
    # Check minimum samples per class
    labels_arr = np.array(labels)
    alert_count = np.sum(labels_arr == 0)
    drowsy_count = np.sum(labels_arr == 1)
    
    if alert_count < min_samples or drowsy_count < min_samples:
        return None, None, alert_count, drowsy_count
    
    # Balance classes
    min_class = min(alert_count, drowsy_count)
    
    alert_indices = np.where(labels_arr == 0)[0]
    drowsy_indices = np.where(labels_arr == 1)[0]
    
    # Select most extreme samples for each class
    # For alert: select samples with lowest PERCLOS
    # For drowsy: select samples with highest PERCLOS
    # Since we don't have PERCLOS values stored, just take random balanced sample
    np.random.seed(42)
    selected_alert = np.random.choice(alert_indices, min_class, replace=False)
    selected_drowsy = np.random.choice(drowsy_indices, min_class, replace=False)
    
    selected_indices = np.concatenate([selected_alert, selected_drowsy])
    np.random.shuffle(selected_indices)
    
    balanced_segments = [segments[i] for i in selected_indices]
    balanced_labels = [labels[i] for i in selected_indices]
    
    return balanced_segments, balanced_labels, alert_count, drowsy_count


def prepare_loso_folds(all_segments, all_labels, all_subject_ids, output_dir, 
                       num_freq_bins=128, dataset_names=None):
    """
    Prepare Leave-One-Subject-Out cross-validation folds.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    unique_subjects = np.unique(all_subject_ids)
    n_subjects = len(unique_subjects)
    n_channels = all_segments[0].shape[0]
    
    print(f"\nPreparing {n_subjects} LOSO folds...")
    print(f"Total samples: {len(all_labels)}")
    print(f"Channels: {n_channels}")
    print(f"PSD frequency bins: {num_freq_bins}")
    
    # Compute PSD features for all data
    print("\nComputing PSD features...")
    all_psd = compute_psd_features(all_segments, fs=128, num_freq_bins=num_freq_bins)
    print(f"PSD shape: {all_psd.shape}")
    print(f"Value range: [{all_psd.min():.2f}, {all_psd.max():.2f}]")
    
    # Convert labels: 0=Alert, 1=Drowsy -> 1=Alert, 2=Drowsy (MATLAB convention)
    labels_matlab = np.array(all_labels) + 1
    subject_ids_arr = np.array(all_subject_ids)
    
    fold_info = []
    
    for fold_idx, test_subject in enumerate(unique_subjects, 1):
        print(f"\n--- Fold {fold_idx}: Test Subject {test_subject} ---")
        
        test_mask = subject_ids_arr == test_subject
        train_mask = ~test_mask
        
        X_test = all_psd[:, :, :, test_mask]
        y_test = labels_matlab[test_mask]
        subj_test = subject_ids_arr[test_mask]
        
        X_train_full = all_psd[:, :, :, train_mask]
        y_train_full = labels_matlab[train_mask]
        subj_train_full = subject_ids_arr[train_mask]
        
        # Split train into train/val (90/10)
        n_train_full = X_train_full.shape[3]
        indices = np.arange(n_train_full)
        
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
            'test_subject': test_subject,
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
        'dataset': 'SEED-VLA+VRW',
        'preprocessing': 'PSD: |FFT|^2/N, log-transformed, normalized per freq bin',
        'sampling_rate': 128
    }
    
    savemat(os.path.join(output_dir, 'metadata.mat'), metadata)
    savemat(os.path.join(output_dir, 'export_complete.mat'), {'complete': True})
    
    # Save summary
    with open(os.path.join(output_dir, 'folds.txt'), 'w') as f:
        f.write("SEED-VLA+VRW LOSO Folds Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total subjects: {n_subjects}\n")
        f.write(f"Total samples: {len(all_labels)}\n")
        f.write(f"Channels: {n_channels}\n")
        f.write(f"PSD frequency bins: {num_freq_bins}\n")
        f.write(f"Preprocessing: PSD = |FFT|^2/N, log-transformed, normalized\n\n")
        
        if dataset_names:
            f.write("Subject mapping:\n")
            for subj_id, ds_name in dataset_names.items():
                f.write(f"  Subject {subj_id}: {ds_name}\n")
            f.write("\n")
        
        for info in fold_info:
            f.write(f"Fold {info['fold']}: Test Subject {info['test_subject']}\n")
            f.write(f"  Train: {info['train_samples']}, Val: {info['val_samples']}, Test: {info['test_samples']}\n")
    
    return fold_info


def main():
    parser = argparse.ArgumentParser(description='Prepare SEED-VLA+VRW for DANN')
    parser.add_argument('--alert_thresh', type=float, default=0.35,
                        help='PERCLOS threshold for alert state (default: 0.35)')
    parser.add_argument('--drowsy_thresh', type=float, default=0.70,
                        help='PERCLOS threshold for drowsy state (default: 0.70)')
    parser.add_argument('--min_samples', type=int, default=50,
                        help='Minimum samples per class per subject (default: 50)')
    parser.add_argument('--output_dir', type=str, default='diagnostics/python_data_vla_vrw',
                        help='Output directory for processed data')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SEED-VLA + SEED-VRW Data Preparation")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Alert threshold: PERCLOS < {args.alert_thresh}")
    print(f"  Drowsy threshold: PERCLOS > {args.drowsy_thresh}")
    print(f"  Minimum samples per class: {args.min_samples}")
    print(f"  Output directory: {args.output_dir}")
    
    # Data paths
    lab_eeg_dir = 'data/VLA_VRW/lab/EEG'
    lab_perclos_dir = 'data/VLA_VRW/lab/perclos'
    real_eeg_dir = 'data/VLA_VRW/real/EEG'
    real_perclos_dir = 'data/VLA_VRW/real/perclos'
    
    all_segments = []
    all_labels = []
    all_subject_ids = []
    dataset_names = {}
    
    subject_counter = 1
    
    # Process VLA (Lab) subjects
    print("\n" + "=" * 60)
    print("Processing VLA (Lab) Dataset")
    print("=" * 60)
    
    lab_files = sorted([f for f in os.listdir(lab_eeg_dir) if f.endswith('.edf')],
                       key=lambda x: int(x.split('.')[0]))
    
    for edf_file in lab_files:
        orig_id = edf_file.split('.')[0]
        eeg_path = os.path.join(lab_eeg_dir, edf_file)
        perclos_path = os.path.join(lab_perclos_dir, f'{orig_id}.mat')
        
        segments, labels, alert_count, drowsy_count = process_subject(
            eeg_path, perclos_path, args.alert_thresh, args.drowsy_thresh, args.min_samples
        )
        
        if segments is None:
            print(f"  VLA Subject {orig_id}: SKIPPED (Alert={alert_count}, Drowsy={drowsy_count})")
            continue
        
        print(f"  VLA Subject {orig_id} -> Subject {subject_counter}: {len(labels)} samples "
              f"(Alert={alert_count}, Drowsy={drowsy_count})")
        
        all_segments.extend(segments)
        all_labels.extend(labels)
        all_subject_ids.extend([subject_counter] * len(labels))
        dataset_names[subject_counter] = f"VLA_{orig_id}"
        subject_counter += 1
    
    # Process VRW (Real) subjects
    print("\n" + "=" * 60)
    print("Processing VRW (Real-World) Dataset")
    print("=" * 60)
    
    real_files = sorted([f for f in os.listdir(real_eeg_dir) if f.endswith('.edf')],
                        key=lambda x: int(x.split('.')[0]))
    
    for edf_file in real_files:
        orig_id = edf_file.split('.')[0]
        eeg_path = os.path.join(real_eeg_dir, edf_file)
        perclos_path = os.path.join(real_perclos_dir, f'{orig_id}.mat')
        
        segments, labels, alert_count, drowsy_count = process_subject(
            eeg_path, perclos_path, args.alert_thresh, args.drowsy_thresh, args.min_samples
        )
        
        if segments is None:
            print(f"  VRW Subject {orig_id}: SKIPPED (Alert={alert_count}, Drowsy={drowsy_count})")
            continue
        
        print(f"  VRW Subject {orig_id} -> Subject {subject_counter}: {len(labels)} samples "
              f"(Alert={alert_count}, Drowsy={drowsy_count})")
        
        all_segments.extend(segments)
        all_labels.extend(labels)
        all_subject_ids.extend([subject_counter] * len(labels))
        dataset_names[subject_counter] = f"VRW_{orig_id}"
        subject_counter += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total valid subjects: {subject_counter - 1}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Alert samples: {sum(1 for l in all_labels if l == 0)}")
    print(f"Drowsy samples: {sum(1 for l in all_labels if l == 1)}")
    
    if len(all_labels) == 0:
        print("\nERROR: No valid subjects found! Try relaxing thresholds.")
        return
    
    # Prepare LOSO folds
    fold_info = prepare_loso_folds(
        all_segments, all_labels, all_subject_ids, 
        args.output_dir, num_freq_bins=128, dataset_names=dataset_names
    )
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Number of folds: {len(fold_info)}")
    print(f"\nTo train the DANN model, run:")
    print(f"  python domain_adversarial_training.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()

