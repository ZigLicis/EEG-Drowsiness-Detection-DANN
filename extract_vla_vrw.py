#!/usr/bin/env python3
"""
Extract SEED-VLA/VRW Dataset for Drowsiness Detection
======================================================

This script extracts EEG samples from the SEED-VLA (lab) and SEED-VRW (real-world)
datasets following the same procedure as SEED-VIG:

1. Downsample EEG to 128 Hz
2. Low-pass filter at 45 Hz (bandpass 1-45 Hz for drowsiness detection)
3. Extract 3-second windows prior to PERCLOS evaluation events
4. Label samples as:
   - 'alert' when PERCLOS < 0.35
   - 'drowsy' when PERCLOS > 0.7
   - Discard middle range
5. Discard sessions with < 50 samples of either class
6. Balance classes by selecting most alert/drowsiest samples

Output format matches SEED-VIG:
- EEGsample: (n_samples, n_channels, 384) - 3s @ 128Hz
- subindex: (n_samples, 1) - subject index
- substate: (n_samples, 1) - 0=alert, 1=drowsy

Usage:
    python extract_vla_vrw.py --dataset lab --output data/VLA_extracted.mat
    python extract_vla_vrw.py --dataset real --output data/VRW_extracted.mat
    python extract_vla_vrw.py --dataset both --output data/VLA_VRW_extracted.mat
"""

import os
import numpy as np
import mne
from scipy.io import loadmat, savemat
from scipy.signal import resample
import warnings
warnings.filterwarnings('ignore')

# Parameters matching SEED-VIG extraction
TARGET_SFREQ = 128  # Target sampling rate
WINDOW_SEC = 3  # Window length in seconds
WINDOW_SAMPLES = TARGET_SFREQ * WINDOW_SEC  # 384 samples

# PERCLOS thresholds
ALERT_THRESHOLD = 0.35
DROWSY_THRESHOLD = 0.7

# Minimum samples per class to include a session
MIN_SAMPLES_PER_CLASS = 50

# EEG channels to use (17 channels matching SEED-VIG style)
# Original channels are referenced to Pz, we'll select standard 10-20 channels
CHANNELS_TO_USE = [
    'EEG Fp1 - Pz', 'EEG Fp2 - Pz',  # Frontal pole
    'EEG F7 - Pz', 'EEG F3 - Pz', 'EEG Fz - Pz', 'EEG F4 - Pz', 'EEG F8 - Pz',  # Frontal
    'EEG T3 - Pz', 'EEG C3 - Pz', 'EEG Cz - Pz', 'EEG C4 - Pz', 'EEG T4 - Pz',  # Central/Temporal
    'EEG T5 - Pz', 'EEG P3 - Pz', 'EEG P4 - Pz', 'EEG T6 - Pz',  # Parietal/Temporal
    'EEG O1 - Pz', 'EEG O2 - Pz'  # Occipital
]

def load_eeg_data(edf_path, target_sfreq=128, low_freq=1, high_freq=45):
    """
    Load and preprocess EEG data from EDF file.
    
    Args:
        edf_path: Path to EDF file
        target_sfreq: Target sampling frequency
        low_freq: Low cutoff for bandpass filter
        high_freq: High cutoff for bandpass filter
        
    Returns:
        data: (n_channels, n_samples) array
        sfreq: Sampling frequency
        ch_names: Channel names
    """
    # Load raw data
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Get original sampling rate
    orig_sfreq = raw.info['sfreq']
    
    # Select channels (only EEG, exclude Trigger)
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG')]
    raw.pick_channels(eeg_channels)
    
    # Bandpass filter (1-45 Hz for drowsiness detection)
    raw.filter(low_freq, high_freq, verbose=False)
    
    # Resample to target frequency
    if orig_sfreq != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    
    # Get data
    data = raw.get_data()
    
    return data, target_sfreq, raw.ch_names

def extract_windows(eeg_data, perclos_values, sfreq, window_sec=3, skip_initial_sec=24):
    """
    Extract EEG windows aligned with PERCLOS evaluations.
    
    The PERCLOS values are typically computed at regular intervals (e.g., every ~8-9 seconds).
    We extract 3-second windows PRIOR to each PERCLOS evaluation.
    
    Args:
        eeg_data: (n_channels, n_samples) array
        perclos_values: (n_evaluations,) array of PERCLOS values
        sfreq: Sampling frequency
        window_sec: Window length in seconds
        skip_initial_sec: Seconds to skip at the beginning
        
    Returns:
        windows: List of (n_channels, window_samples) arrays
        labels: List of PERCLOS values for each window
    """
    n_channels, n_samples = eeg_data.shape
    n_perclos = len(perclos_values)
    window_samples = int(window_sec * sfreq)
    
    # Calculate the time interval between PERCLOS evaluations
    # Total duration minus initial skip, divided by number of evaluations
    total_duration_sec = n_samples / sfreq
    usable_duration_sec = total_duration_sec - skip_initial_sec
    
    # PERCLOS is typically evaluated every ~8-9 seconds
    perclos_interval_sec = usable_duration_sec / n_perclos
    
    windows = []
    labels = []
    
    for i in range(n_perclos):
        # Calculate the time of this PERCLOS evaluation
        eval_time_sec = skip_initial_sec + (i + 1) * perclos_interval_sec
        
        # Extract window PRIOR to evaluation
        window_end_sample = int(eval_time_sec * sfreq)
        window_start_sample = window_end_sample - window_samples
        
        # Check bounds
        if window_start_sample < 0 or window_end_sample > n_samples:
            continue
        
        # Extract window
        window = eeg_data[:, window_start_sample:window_end_sample]
        
        # Verify window size
        if window.shape[1] != window_samples:
            continue
        
        windows.append(window)
        labels.append(perclos_values[i])
    
    return windows, labels

def process_dataset(data_dir, dataset_name='lab'):
    """
    Process all subjects in a dataset (lab or real).
    
    Args:
        data_dir: Base directory containing EEG/ and perclos/ subdirectories
        dataset_name: 'lab' or 'real' for identification
        
    Returns:
        all_samples: List of (n_channels, 384) arrays
        all_labels: List of 0 (alert) or 1 (drowsy)
        all_subjects: List of subject indices
    """
    eeg_dir = os.path.join(data_dir, 'EEG')
    perclos_dir = os.path.join(data_dir, 'perclos')
    
    # Find all subjects
    eeg_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith('.edf')])
    
    all_samples = []
    all_labels = []
    all_subjects = []
    
    valid_subject_idx = 0
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    for eeg_file in eeg_files:
        subject_id = eeg_file.replace('.edf', '')
        eeg_path = os.path.join(eeg_dir, eeg_file)
        perclos_path = os.path.join(perclos_dir, f'{subject_id}.mat')
        
        if not os.path.exists(perclos_path):
            print(f"Subject {subject_id}: PERCLOS file not found, skipping")
            continue
        
        print(f"\nSubject {subject_id}:")
        
        # Load EEG data
        try:
            eeg_data, sfreq, ch_names = load_eeg_data(eeg_path, TARGET_SFREQ)
            print(f"  EEG: {eeg_data.shape[1]/sfreq:.1f}s, {eeg_data.shape[0]} channels @ {sfreq}Hz")
        except Exception as e:
            print(f"  Error loading EEG: {e}")
            continue
        
        # Load PERCLOS
        try:
            perclos_data = loadmat(perclos_path)
            perclos_values = perclos_data['perclos'].flatten()
            print(f"  PERCLOS: {len(perclos_values)} evaluations")
        except Exception as e:
            print(f"  Error loading PERCLOS: {e}")
            continue
        
        # Extract windows
        windows, perclos_labels = extract_windows(eeg_data, perclos_values, sfreq)
        print(f"  Extracted {len(windows)} windows")
        
        if len(windows) == 0:
            print(f"  No valid windows, skipping")
            continue
        
        # Classify based on PERCLOS thresholds
        alert_windows = []
        alert_perclos = []
        drowsy_windows = []
        drowsy_perclos = []
        
        for window, perclos in zip(windows, perclos_labels):
            if perclos < ALERT_THRESHOLD:
                alert_windows.append(window)
                alert_perclos.append(perclos)
            elif perclos > DROWSY_THRESHOLD:
                drowsy_windows.append(window)
                drowsy_perclos.append(perclos)
        
        n_alert = len(alert_windows)
        n_drowsy = len(drowsy_windows)
        print(f"  Alert (<{ALERT_THRESHOLD}): {n_alert}, Drowsy (>{DROWSY_THRESHOLD}): {n_drowsy}")
        
        # Check minimum samples requirement
        if n_alert < MIN_SAMPLES_PER_CLASS or n_drowsy < MIN_SAMPLES_PER_CLASS:
            print(f"  Insufficient samples (need {MIN_SAMPLES_PER_CLASS} per class), skipping")
            continue
        
        # Balance classes by selecting most extreme samples
        n_samples_per_class = min(n_alert, n_drowsy)
        
        # Sort alert by PERCLOS (ascending - most alert first)
        alert_sorted_idx = np.argsort(alert_perclos)
        selected_alert = [alert_windows[i] for i in alert_sorted_idx[:n_samples_per_class]]
        
        # Sort drowsy by PERCLOS (descending - most drowsy first)
        drowsy_sorted_idx = np.argsort(drowsy_perclos)[::-1]
        selected_drowsy = [drowsy_windows[i] for i in drowsy_sorted_idx[:n_samples_per_class]]
        
        print(f"  Selected {n_samples_per_class} per class (balanced)")
        
        # Add to dataset
        for window in selected_alert:
            all_samples.append(window)
            all_labels.append(0)  # Alert
            all_subjects.append(valid_subject_idx)
        
        for window in selected_drowsy:
            all_samples.append(window)
            all_labels.append(1)  # Drowsy
            all_subjects.append(valid_subject_idx)
        
        valid_subject_idx += 1
    
    print(f"\n{dataset_name.upper()} Summary:")
    print(f"  Valid subjects: {valid_subject_idx}")
    print(f"  Total samples: {len(all_samples)}")
    if len(all_labels) > 0:
        print(f"  Alert: {sum(1 for l in all_labels if l == 0)}")
        print(f"  Drowsy: {sum(1 for l in all_labels if l == 1)}")
    
    return all_samples, all_labels, all_subjects, valid_subject_idx

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract SEED-VLA/VRW dataset')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['lab', 'real', 'both'],
                        help='Which dataset to process')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .mat file path')
    parser.add_argument('--data_dir', type=str, default='data/VLA_VRW',
                        help='Base directory for VLA_VRW data')
    parser.add_argument('--min_samples', type=int, default=50,
                        help='Minimum samples per class to include a session')
    parser.add_argument('--alert_threshold', type=float, default=0.35,
                        help='PERCLOS threshold for alert (default: 0.35)')
    parser.add_argument('--drowsy_threshold', type=float, default=0.7,
                        help='PERCLOS threshold for drowsy (default: 0.7)')
    
    args = parser.parse_args()
    
    # Update global thresholds
    global ALERT_THRESHOLD, DROWSY_THRESHOLD, MIN_SAMPLES_PER_CLASS
    ALERT_THRESHOLD = args.alert_threshold
    DROWSY_THRESHOLD = args.drowsy_threshold
    MIN_SAMPLES_PER_CLASS = args.min_samples
    
    all_samples = []
    all_labels = []
    all_subjects = []
    subject_offset = 0
    
    if args.dataset in ['lab', 'both']:
        lab_dir = os.path.join(args.data_dir, 'lab')
        samples, labels, subjects, n_subjects = process_dataset(lab_dir, 'lab')
        
        # Add with subject offset
        all_samples.extend(samples)
        all_labels.extend(labels)
        all_subjects.extend([s + subject_offset for s in subjects])
        subject_offset += n_subjects
    
    if args.dataset in ['real', 'both']:
        real_dir = os.path.join(args.data_dir, 'real')
        samples, labels, subjects, n_subjects = process_dataset(real_dir, 'real')
        
        # Add with subject offset
        all_samples.extend(samples)
        all_labels.extend(labels)
        all_subjects.extend([s + subject_offset for s in subjects])
        subject_offset += n_subjects
    
    if len(all_samples) == 0:
        print("\nNo valid samples extracted!")
        print("Try relaxing thresholds:")
        print("  --alert_threshold 0.4 --drowsy_threshold 0.6 --min_samples 30")
        return
    
    # Convert to arrays
    # Stack samples: (n_samples, n_channels, n_timepoints)
    EEGsample = np.stack(all_samples, axis=0)
    substate = np.array(all_labels).reshape(-1, 1)
    subindex = np.array(all_subjects).reshape(-1, 1) + 1  # 1-based indexing
    
    print(f"\n{'='*60}")
    print("FINAL DATASET")
    print(f"{'='*60}")
    print(f"EEGsample shape: {EEGsample.shape}")
    print(f"  - {EEGsample.shape[0]} samples")
    print(f"  - {EEGsample.shape[1]} channels")
    print(f"  - {EEGsample.shape[2]} timepoints ({EEGsample.shape[2]/TARGET_SFREQ}s @ {TARGET_SFREQ}Hz)")
    print(f"substate: {substate.shape} (0=alert, 1=drowsy)")
    print(f"subindex: {subindex.shape} (subjects 1-{subject_offset})")
    print(f"Total subjects: {subject_offset}")
    print(f"Alert samples: {np.sum(substate == 0)}")
    print(f"Drowsy samples: {np.sum(substate == 1)}")
    
    # Determine output path
    if args.output is None:
        if args.dataset == 'lab':
            output_path = 'data/VLA_extracted.mat'
        elif args.dataset == 'real':
            output_path = 'data/VRW_extracted.mat'
        else:
            output_path = 'data/VLA_VRW_extracted.mat'
    else:
        output_path = args.output
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    savemat(output_path, {
        'EEGsample': EEGsample,
        'substate': substate,
        'subindex': subindex
    }, do_compression=True)
    
    print(f"\nSaved to: {output_path}")
    
    # Print per-subject breakdown
    print(f"\nPer-subject breakdown:")
    for s in range(1, subject_offset + 1):
        mask = subindex.flatten() == s
        n_total = np.sum(mask)
        n_alert = np.sum((subindex.flatten() == s) & (substate.flatten() == 0))
        n_drowsy = np.sum((subindex.flatten() == s) & (substate.flatten() == 1))
        print(f"  Subject {s}: {n_total} samples (alert={n_alert}, drowsy={n_drowsy})")

if __name__ == "__main__":
    main()

