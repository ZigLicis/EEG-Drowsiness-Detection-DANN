#!/usr/bin/env python3
"""
Domain Adversarial Neural Network (DANN) for Combined SADT + SEED-VIG Drowsiness Detection
===========================================================================================

This script implements a Domain Adversarial CNN with Gradient Reversal Layer (GRL)
for cross-subject drowsiness detection using BOTH SADT and SEED-VIG datasets.

The key idea: Use adversarial training to learn features that are:
  1. Discriminative for drowsiness detection (main task)
  2. Invariant to subject identity (via gradient reversal on subject classifier)

This forces the network to learn universal drowsiness patterns rather than
subject-specific signatures, improving generalization to unseen subjects.

Datasets:
  - SADT: 2022 samples, 30 channels, 11 subjects (3s @ 128Hz)
  - SEED-VIG: 4566 samples, 17 channels, 12 subjects (3s @ 128Hz)
  - Combined: 6588 samples, 8 common channels, 23 subjects

Common Channels (8): FT7, FT8, TP7, TP8, PZ, O1, OZ, O2

Architecture:
  - Feature Extractor: CNN backbone that extracts EEG features
  - Drowsiness Classifier: Predicts alert/drowsy (main task)
  - Subject Classifier: Predicts subject ID (adversarial task with GRL)

The GRL reverses gradients during backpropagation, so minimizing subject
classification loss actually maximizes confusion - making features subject-invariant.

Usage:
    python train_dann_combined.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# ============================================================================
# CHANNEL CONFIGURATION
# ============================================================================

# Common 8 channels between SADT (30ch) and SEED-VIG (17ch)
COMMON_CHANNELS = ['FT7', 'FT8', 'TP7', 'TP8', 'PZ', 'O1', 'OZ', 'O2']

# SADT 30-channel order (standard 10-20 extended montage)
SADT_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
    'FC4', 'FT8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'TP7', 'CP3', 'CPZ',
    'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2'
]

# SEED-VIG 17-channel order (temporal + posterior)
SEED_VIG_CHANNELS = [
    'FT7', 'T7', 'TP7', 'FT8', 'T8', 'TP8',  # Temporal (0-5)
    'P7', 'P3', 'PZ', 'P4', 'P8',             # Parietal (6-10)
    'O1', 'OZ', 'O2',                          # Occipital (11-13)
    'PO7', 'POZ', 'PO8'                        # Parieto-occipital (14-16)
]


def get_channel_indices(channel_list, target_channels):
    """Get indices of target channels in a channel list (case-insensitive)"""
    channel_upper = [ch.upper() for ch in channel_list]
    indices = []
    for ch in target_channels:
        if ch.upper() in channel_upper:
            indices.append(channel_upper.index(ch.upper()))
        else:
            raise ValueError(f"Channel {ch} not found in channel list")
    return indices


# Pre-compute channel indices
SADT_CHANNEL_INDICES = get_channel_indices(SADT_CHANNELS, COMMON_CHANNELS)
SEED_VIG_CHANNEL_INDICES = get_channel_indices(SEED_VIG_CHANNELS, COMMON_CHANNELS)


# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for Domain Adversarial Training.
    
    Forward pass: Identity function (x -> x)
    Backward pass: Negates and scales gradients (grad -> -lambda * grad)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for the gradient reversal function."""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


# ============================================================================
# FEATURE EXTRACTOR (CNN BACKBONE)
# ============================================================================

class FeatureExtractor(nn.Module):
    """
    CNN Feature Extractor for raw EEG signals.
    
    Architecture inspired by EEGNet with modifications for multi-channel input.
    
    Input: (batch, 1, channels=8, time=384)
    Output: (batch, feature_dim)
    """
    def __init__(self, num_channels=8, num_timepoints=384, dropout=0.5):
        super().__init__()
        
        # Temporal convolution - learns frequency filters
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True)
        )
        
        # Spatial convolution - learns channel combinations (depthwise)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(num_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        # Separable convolution - refines temporal patterns
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        # Additional conv block for deeper features
        self.deep_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        # Calculate feature dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_timepoints)
            dummy = self._forward_conv(dummy)
            self.feature_dim = dummy.numel()
        
        # Feature projection layer
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.output_dim = 256
        
    def _forward_conv(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.deep_conv(x)
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# CLASSIFIERS
# ============================================================================

class DrowsinessClassifier(nn.Module):
    """Drowsiness classifier head - main task."""
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class SubjectClassifier(nn.Module):
    """Subject classifier head - adversarial task with GRL."""
    def __init__(self, input_dim=256, hidden_dim=128, num_subjects=23, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_subjects)
        )
    
    def forward(self, x):
        return self.classifier(x)


# ============================================================================
# DOMAIN ADVERSARIAL NEURAL NETWORK
# ============================================================================

class DANN(nn.Module):
    """
    Domain Adversarial Neural Network for Cross-Subject EEG Classification.
    
    Architecture:
        Input -> Feature Extractor -> [features]
                                          |
                          +---------------+---------------+
                          |                               |
                          v                               v
                  Drowsiness Classifier           GRL -> Subject Classifier
                  (minimize loss)                 (maximize confusion)
    """
    def __init__(self, num_channels=8, num_timepoints=384, 
                 num_subjects=23, dropout=0.5):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            num_channels=num_channels,
            num_timepoints=num_timepoints,
            dropout=dropout
        )
        
        feature_dim = self.feature_extractor.output_dim
        
        self.drowsiness_classifier = DrowsinessClassifier(
            input_dim=feature_dim,
            num_classes=2,
            dropout=dropout * 0.6
        )
        
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        self.subject_classifier = SubjectClassifier(
            input_dim=feature_dim,
            num_subjects=num_subjects,
            dropout=dropout * 0.6
        )
        
        self.num_subjects = num_subjects
        
    def forward(self, x, lambda_grl=1.0):
        """
        Forward pass.
        
        Args:
            x: Input EEG data (batch, 1, channels, time)
            lambda_grl: Scaling factor for gradient reversal (0 to 1)
                       
        Returns:
            drowsiness_logits: Drowsiness predictions (batch, 2)
            subject_logits: Subject predictions (batch, num_subjects)
            features: Extracted features (batch, feature_dim)
        """
        features = self.feature_extractor(x)
        drowsiness_logits = self.drowsiness_classifier(features)
        
        self.grl.set_lambda(lambda_grl)
        reversed_features = self.grl(features)
        subject_logits = self.subject_classifier(reversed_features)
        
        return drowsiness_logits, subject_logits, features
    
    def predict_drowsiness(self, x):
        """Predict drowsiness only (for inference)."""
        features = self.feature_extractor(x)
        return self.drowsiness_classifier(features)


# ============================================================================
# DATASET
# ============================================================================

class CombinedEEGDataset(Dataset):
    """
    PyTorch Dataset for combined SADT + SEED-VIG EEG data.
    """
    def __init__(self, X, y, subject_ids, dataset_ids, normalize=True, stats=None):
        """
        Args:
            X: EEG data (samples, channels, time)
            y: Drowsiness labels (0=alert, 1=drowsy)
            subject_ids: Global subject IDs (0-indexed, 0-22)
            dataset_ids: Dataset source (0=SADT, 1=SEED-VIG)
            normalize: Whether to apply z-score normalization
            stats: (mean, std) tuple for normalization
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.subject_ids = subject_ids.astype(np.int64)
        self.dataset_ids = dataset_ids.astype(np.int64)
        
        # Z-score normalization per channel
        if normalize:
            if stats is None:
                self.mean = self.X.mean(axis=(0, 2), keepdims=True)
                self.std = self.X.std(axis=(0, 2), keepdims=True) + 1e-8
            else:
                self.mean, self.std = stats
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean, self.std = None, None
        
        # Add channel dimension: (samples, channels, time) -> (samples, 1, channels, time)
        self.X = self.X[:, np.newaxis, :, :]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.subject_ids[idx]),
            torch.tensor(self.dataset_ids[idx])
        )
    
    def get_stats(self):
        return (self.mean, self.std)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_combined_datasets(data_dir='data'):
    """
    Load SADT and SEED-VIG datasets, select common channels, and combine.
    
    Returns:
        X: Combined EEG data (samples, 8 channels, 384 time points)
        y: Drowsiness labels (0=alert, 1=drowsy)
        subject_ids: Global subject IDs (0-indexed: SADT 0-10, SEED-VIG 11-22)
        dataset_ids: Dataset source (0=SADT, 1=SEED-VIG)
        subject_info: Dict with subject metadata
    """
    print("=" * 60)
    print("Loading and combining datasets")
    print("=" * 60)
    
    # Load SADT
    print("\n[SADT Dataset]")
    sadt_path = os.path.join(data_dir, 'SADT', 'sadt.mat')
    sadt = loadmat(sadt_path)
    
    sadt_X = sadt['EEGsample']  # (2022, 30, 384)
    sadt_y = sadt['substate'].flatten()  # (2022,)
    sadt_subjects = sadt['subindex'].flatten() - 1  # Convert to 0-indexed (0-10)
    
    # Select common channels for SADT
    sadt_X = sadt_X[:, SADT_CHANNEL_INDICES, :]  # (2022, 8, 384)
    
    print(f"  Samples: {sadt_X.shape[0]}")
    print(f"  Original channels: 30 -> Common channels: {sadt_X.shape[1]}")
    print(f"  Subjects: {len(np.unique(sadt_subjects))} (IDs: 0-10)")
    print(f"  Labels: Alert={np.sum(sadt_y==0)}, Drowsy={np.sum(sadt_y==1)}")
    
    # Load SEED-VIG
    print("\n[SEED-VIG Dataset]")
    seed_path = os.path.join(data_dir, 'SEED-VIG_Extracted', 'SEED_VIG.mat')
    seed = loadmat(seed_path)
    
    seed_X = seed['EEGsample']  # (4566, 17, 384)
    seed_y = seed['substate'].flatten()  # (4566,)
    seed_subjects = seed['subindex'].flatten() - 1 + 11  # Offset to 11-22
    
    # Select common channels for SEED-VIG
    seed_X = seed_X[:, SEED_VIG_CHANNEL_INDICES, :]  # (4566, 8, 384)
    
    print(f"  Samples: {seed_X.shape[0]}")
    print(f"  Original channels: 17 -> Common channels: {seed_X.shape[1]}")
    print(f"  Subjects: {len(np.unique(seed_subjects))} (IDs: 11-22)")
    print(f"  Labels: Alert={np.sum(seed_y==0)}, Drowsy={np.sum(seed_y==1)}")
    
    # Combine datasets
    X = np.concatenate([sadt_X, seed_X], axis=0)
    y = np.concatenate([sadt_y, seed_y], axis=0)
    subject_ids = np.concatenate([sadt_subjects, seed_subjects], axis=0)
    dataset_ids = np.concatenate([
        np.zeros(len(sadt_y), dtype=np.int64),  # SADT = 0
        np.ones(len(seed_y), dtype=np.int64)    # SEED-VIG = 1
    ], axis=0)
    
    # Subject info for reporting
    subject_info = {}
    for subj_id in np.unique(subject_ids):
        mask = subject_ids == subj_id
        dataset = "SADT" if subj_id < 11 else "SEED-VIG"
        subject_info[subj_id] = {
            'dataset': dataset,
            'n_samples': np.sum(mask),
            'n_alert': np.sum(y[mask] == 0),
            'n_drowsy': np.sum(y[mask] == 1)
        }
    
    print("\n[Combined Dataset]")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Common channels: {X.shape[1]} ({', '.join(COMMON_CHANNELS)})")
    print(f"  Time points: {X.shape[2]} (3s @ 128Hz)")
    print(f"  Total subjects: {len(np.unique(subject_ids))}")
    print(f"    - SADT: 11 subjects (IDs 0-10)")
    print(f"    - SEED-VIG: 12 subjects (IDs 11-22)")
    print(f"  Total labels: Alert={np.sum(y==0)}, Drowsy={np.sum(y==1)}")
    
    return X, y, subject_ids, dataset_ids, subject_info


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_lambda_schedule(epoch, num_epochs, schedule='sigmoid'):
    """Compute lambda for gradient reversal layer."""
    p = epoch / num_epochs
    if schedule == 'constant':
        return 1.0
    elif schedule == 'linear':
        return p
    elif schedule == 'sigmoid':
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs,
                alpha_subject=0.5, lambda_schedule='sigmoid'):
    """Train for one epoch."""
    model.train()
    
    lambda_grl = compute_lambda_schedule(epoch, num_epochs, lambda_schedule)
    
    drowsiness_criterion = nn.CrossEntropyLoss()
    subject_criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_drowsiness_loss = 0
    total_subject_loss = 0
    drowsiness_correct = 0
    subject_correct = 0
    total_samples = 0
    
    for data, drowsiness_labels, subject_labels, _ in train_loader:
        data = data.to(device)
        drowsiness_labels = drowsiness_labels.to(device)
        subject_labels = subject_labels.to(device)
        
        optimizer.zero_grad()
        
        drowsiness_logits, subject_logits, _ = model(data, lambda_grl)
        
        loss_drowsiness = drowsiness_criterion(drowsiness_logits, drowsiness_labels)
        loss_subject = subject_criterion(subject_logits, subject_labels)
        
        loss = loss_drowsiness + alpha_subject * loss_subject
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_drowsiness_loss += loss_drowsiness.item() * data.size(0)
        total_subject_loss += loss_subject.item() * data.size(0)
        
        _, drowsiness_pred = drowsiness_logits.max(1)
        _, subject_pred = subject_logits.max(1)
        
        drowsiness_correct += drowsiness_pred.eq(drowsiness_labels).sum().item()
        subject_correct += subject_pred.eq(subject_labels).sum().item()
        total_samples += data.size(0)
    
    return (
        total_loss / total_samples,
        total_drowsiness_loss / total_samples,
        total_subject_loss / total_samples,
        100.0 * drowsiness_correct / total_samples,
        100.0 * subject_correct / total_samples,
        lambda_grl
    )


def evaluate(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    all_subject_preds = []
    all_subject_labels = []
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, drowsiness_labels, subject_labels, _ in loader:
            data = data.to(device)
            drowsiness_labels = drowsiness_labels.to(device)
            
            drowsiness_logits, subject_logits, _ = model(data, lambda_grl=0.0)
            
            loss = criterion(drowsiness_logits, drowsiness_labels)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            _, drowsiness_pred = drowsiness_logits.max(1)
            _, subject_pred = subject_logits.max(1)
            
            all_preds.extend(drowsiness_pred.cpu().numpy())
            all_labels.extend(drowsiness_labels.cpu().numpy())
            all_subject_preds.extend(subject_pred.cpu().numpy())
            all_subject_labels.extend(subject_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_subject_preds = np.array(all_subject_preds)
    all_subject_labels = np.array(all_subject_labels)
    
    accuracy = 100.0 * np.mean(all_preds == all_labels)
    balanced_acc = 100.0 * balanced_accuracy_score(all_labels, all_preds)
    subject_acc = 100.0 * np.mean(all_subject_preds == all_subject_labels)
    
    return (
        total_loss / total_samples,
        accuracy,
        balanced_acc,
        subject_acc,
        all_preds,
        all_labels
    )


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 70)
    print("Domain Adversarial CNN - Combined SADT + SEED-VIG")
    print("=" * 70)
    print("\nUsing Gradient Reversal Layer to learn subject-invariant features")
    print("across 23 subjects from two different datasets\n")
    
    # ==================== Configuration ====================
    data_dir = 'data'
    output_dir = 'results_dann_combined'
    
    # Training hyperparameters
    num_epochs = 150
    batch_size = 64
    learning_rate = 5e-4
    weight_decay = 1e-4
    patience = 20
    
    # DANN hyperparameters
    alpha_subject = 0.5  # Weight for subject adversarial loss
    lambda_schedule = 'sigmoid'
    dropout = 0.5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Load Data ====================
    X, y, subject_ids, dataset_ids, subject_info = load_combined_datasets(data_dir)
    
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    print(f"\nPerforming Leave-One-Subject-Out (LOSO) cross-validation")
    print(f"across all {n_subjects} subjects from both datasets\n")
    
    # ==================== Results Storage ====================
    all_results = []
    fold_accuracies = []
    fold_balanced_accs = []
    fold_subject_accs = []
    
    # ==================== LOSO Cross-Validation ====================
    for fold_idx, test_subject in enumerate(unique_subjects):
        info = subject_info[test_subject]
        dataset_name = info['dataset']
        
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1}/{n_subjects}: Test Subject = {test_subject} ({dataset_name})")
        print(f"{'='*70}")
        
        # Split data
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        subj_train = subject_ids[train_mask]
        subj_test = subject_ids[test_mask]
        dataset_train = dataset_ids[train_mask]
        dataset_test = dataset_ids[test_mask]
        
        # Create validation split (10% of training data)
        n_train = len(y_train)
        indices = np.random.permutation(n_train)
        val_size = int(0.1 * n_train)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_val, y_val = X_train[val_indices], y_train[val_indices]
        subj_val = subj_train[val_indices]
        dataset_val = dataset_train[val_indices]
        
        X_train, y_train = X_train[train_indices], y_train[train_indices]
        subj_train = subj_train[train_indices]
        dataset_train = dataset_train[train_indices]
        
        print(f"Train: {len(y_train)} samples (Alert: {np.sum(y_train==0)}, Drowsy: {np.sum(y_train==1)})")
        print(f"Val: {len(y_val)} samples")
        print(f"Test: {len(y_test)} samples (Alert: {np.sum(y_test==0)}, Drowsy: {np.sum(y_test==1)})")
        
        # Create datasets
        train_dataset = CombinedEEGDataset(X_train, y_train, subj_train, dataset_train, normalize=True)
        stats = train_dataset.get_stats()
        val_dataset = CombinedEEGDataset(X_val, y_val, subj_val, dataset_val, normalize=True, stats=stats)
        test_dataset = CombinedEEGDataset(X_test, y_test, subj_test, dataset_test, normalize=True, stats=stats)
        
        # Class-balanced sampling
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = DANN(
            num_channels=8,  # Common channels
            num_timepoints=384,
            num_subjects=n_subjects,
            dropout=dropout
        ).to(device)
        
        # Count parameters (only on first fold)
        if fold_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )
        
        # Training loop
        best_val_balanced_acc = 0
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_acc': [], 'val_bal_acc': [], 'subject_acc': [], 'lambda': []}
        
        for epoch in range(num_epochs):
            train_loss, drowsiness_loss, subject_loss, train_acc, train_subj_acc, lambda_grl = train_epoch(
                model, train_loader, optimizer, device, epoch, num_epochs,
                alpha_subject=alpha_subject, lambda_schedule=lambda_schedule
            )
            
            val_loss, val_acc, val_bal_acc, val_subj_acc, _, _ = evaluate(model, val_loader, device)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_bal_acc'].append(val_bal_acc)
            history['subject_acc'].append(val_subj_acc)
            history['lambda'].append(lambda_grl)
            
            if val_bal_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_bal_acc
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Loss={train_loss:.4f} | "
                      f"Train Acc={train_acc:.1f}% | Val Bal Acc={val_bal_acc:.1f}% | "
                      f"Subj Acc={val_subj_acc:.1f}% | λ={lambda_grl:.3f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)
        
        test_loss, test_acc, test_bal_acc, test_subj_acc, preds, labels = evaluate(model, test_loader, device)
        
        print(f"\nFold {fold_idx + 1} Results ({dataset_name}):")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Balanced Accuracy: {test_bal_acc:.2f}%")
        print(f"  Subject Classifier Accuracy: {test_subj_acc:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        print(f"  Confusion Matrix:")
        print(f"    Alert  -> Alert: {cm[0,0]:3d}, Drowsy: {cm[0,1]:3d}")
        print(f"    Drowsy -> Alert: {cm[1,0]:3d}, Drowsy: {cm[1,1]:3d}")
        
        # Store results
        fold_accuracies.append(test_acc)
        fold_balanced_accs.append(test_bal_acc)
        fold_subject_accs.append(test_subj_acc)
        
        all_results.append({
            'fold': fold_idx + 1,
            'test_subject': int(test_subject),
            'dataset': dataset_name,
            'test_accuracy': test_acc,
            'balanced_accuracy': test_bal_acc,
            'subject_accuracy': test_subj_acc,
            'predictions': preds,
            'labels': labels,
            'history': history
        })
        
        # Save model for this fold
        torch.save({
            'model_state_dict': best_model_state,
            'fold': fold_idx + 1,
            'test_subject': int(test_subject),
            'dataset': dataset_name,
            'test_accuracy': test_acc,
            'balanced_accuracy': test_bal_acc
        }, os.path.join(output_dir, f'model_fold{fold_idx+1}.pt'))
    
    # ==================== Final Results ====================
    print("\n" + "=" * 70)
    print("FINAL RESULTS - Domain Adversarial CNN (Combined Datasets)")
    print("=" * 70)
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_bal_acc = np.mean(fold_balanced_accs)
    std_bal_acc = np.std(fold_balanced_accs)
    mean_subj_acc = np.mean(fold_subject_accs)
    
    print(f"\nOverall Results:")
    print(f"  Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Mean Balanced Accuracy: {mean_bal_acc:.2f}% ± {std_bal_acc:.2f}%")
    print(f"  Mean Subject Classifier Accuracy: {mean_subj_acc:.2f}% (chance = {100/n_subjects:.1f}%)")
    
    # Separate results by dataset
    sadt_indices = [i for i, s in enumerate(unique_subjects) if s < 11]
    seedvig_indices = [i for i, s in enumerate(unique_subjects) if s >= 11]
    
    sadt_accs = [fold_accuracies[i] for i in sadt_indices]
    sadt_bal_accs = [fold_balanced_accs[i] for i in sadt_indices]
    seedvig_accs = [fold_accuracies[i] for i in seedvig_indices]
    seedvig_bal_accs = [fold_balanced_accs[i] for i in seedvig_indices]
    
    print(f"\nBy Dataset:")
    print(f"  SADT (11 subjects):     {np.mean(sadt_bal_accs):.2f}% ± {np.std(sadt_bal_accs):.2f}% balanced acc")
    print(f"  SEED-VIG (12 subjects): {np.mean(seedvig_bal_accs):.2f}% ± {np.std(seedvig_bal_accs):.2f}% balanced acc")
    
    print("\nPer-fold results:")
    print(f"{'Fold':>4} {'Subject':>7} {'Dataset':>10} {'Accuracy':>10} {'Balanced':>10} {'Subj Acc':>10}")
    print("-" * 60)
    for i, (acc, bal_acc, subj_acc) in enumerate(zip(fold_accuracies, fold_balanced_accs, fold_subject_accs)):
        dataset = "SADT" if unique_subjects[i] < 11 else "SEED-VIG"
        print(f"{i+1:>4} {int(unique_subjects[i]):>7} {dataset:>10} {acc:>9.2f}% {bal_acc:>9.2f}% {subj_acc:>9.2f}%")
    
    # ==================== Save Results ====================
    results_file = os.path.join(output_dir, 'loso_results.mat')
    savemat(results_file, {
        'fold_accuracies': fold_accuracies,
        'fold_balanced_accs': fold_balanced_accs,
        'fold_subject_accs': fold_subject_accs,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_balanced_accuracy': mean_bal_acc,
        'std_balanced_accuracy': std_bal_acc,
        'unique_subjects': unique_subjects,
        'alpha_subject': alpha_subject,
        'lambda_schedule': lambda_schedule,
        'common_channels': COMMON_CHANNELS,
        'sadt_balanced_acc': np.mean(sadt_bal_accs),
        'seedvig_balanced_acc': np.mean(seedvig_bal_accs)
    })
    print(f"\nResults saved to {results_file}")
    
    # ==================== Plot Results ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy per fold (colored by dataset)
    ax1 = axes[0, 0]
    x = np.arange(1, n_subjects + 1)
    colors = ['steelblue' if s < 11 else 'coral' for s in unique_subjects]
    bars = ax1.bar(x, fold_balanced_accs, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=mean_bal_acc, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_bal_acc:.1f}%')
    ax1.axhline(y=np.mean(sadt_bal_accs), color='steelblue', linestyle=':', alpha=0.7, label=f'SADT Mean: {np.mean(sadt_bal_accs):.1f}%')
    ax1.axhline(y=np.mean(seedvig_bal_accs), color='coral', linestyle=':', alpha=0.7, label=f'SEED-VIG Mean: {np.mean(seedvig_bal_accs):.1f}%')
    ax1.set_xlabel('Fold (Test Subject)')
    ax1.set_ylabel('Balanced Accuracy (%)')
    ax1.set_title('LOSO Cross-Validation Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{int(s)}' for s in unique_subjects], rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add dataset labels
    for i, (bar, s) in enumerate(zip(bars, unique_subjects)):
        dataset = 'S' if s < 11 else 'V'
        ax1.text(bar.get_x() + bar.get_width()/2, 5, dataset, 
                ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
    
    # Plot 2: Subject classifier accuracy
    ax2 = axes[0, 1]
    ax2.bar(x, fold_subject_accs, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=100/n_subjects, color='red', linestyle='--', linewidth=2, label=f'Chance ({100/n_subjects:.1f}%)')
    ax2.axhline(y=mean_subj_acc, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean ({mean_subj_acc:.1f}%)')
    ax2.set_xlabel('Fold (Test Subject)')
    ax2.set_ylabel('Subject Classification Accuracy (%)')
    ax2.set_title('Subject Invariance (Lower = Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'S{int(s)}' for s in unique_subjects], rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, max(fold_subject_accs) * 1.2])
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Dataset comparison boxplot
    ax3 = axes[1, 0]
    bp = ax3.boxplot([sadt_bal_accs, seedvig_bal_accs], 
                     labels=['SADT\n(11 subjects)', 'SEED-VIG\n(12 subjects)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax3.set_ylabel('Balanced Accuracy (%)')
    ax3.set_title('Performance by Dataset')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add individual points
    for i, (data, color) in enumerate([(sadt_bal_accs, 'steelblue'), (seedvig_bal_accs, 'coral')]):
        x_jitter = np.random.normal(i+1, 0.04, len(data))
        ax3.scatter(x_jitter, data, color=color, alpha=0.6, edgecolor='black', linewidth=0.5, s=50)
    
    # Plot 4: Training curves from last fold
    ax4 = axes[1, 1]
    last_history = all_results[-1]['history']
    epochs = range(1, len(last_history['train_loss']) + 1)
    ax4.plot(epochs, last_history['val_bal_acc'], label='Val Balanced Acc', color='coral', linewidth=2)
    ax4.plot(epochs, last_history['subject_acc'], label='Subject Acc', color='mediumseagreen', linewidth=2)
    
    # Add lambda on secondary axis
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, last_history['lambda'], label='λ (GRL)', color='purple', linestyle='--', alpha=0.7)
    ax4_twin.set_ylabel('λ (GRL strength)', color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')
    ax4_twin.set_ylim([0, 1.1])
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title(f'Training Curves (Fold {n_subjects})')
    ax4.legend(loc='upper left')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dann_combined_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, 'dann_combined_results.png')}")
    
    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Domain Adversarial CNN trained on combined SADT + SEED-VIG datasets
using gradient reversal to learn subject-invariant drowsiness features.

Configuration:
  - Common channels: {len(COMMON_CHANNELS)} ({', '.join(COMMON_CHANNELS)})
  - Total subjects: {n_subjects} (SADT: 11, SEED-VIG: 12)
  - Total samples: {len(y)}
  - Adversarial weight (α): {alpha_subject}
  - Lambda schedule: {lambda_schedule}

Key Results:
  - Overall Balanced Accuracy: {mean_bal_acc:.1f}% ± {std_bal_acc:.1f}%
  - SADT subjects: {np.mean(sadt_bal_accs):.1f}% ± {np.std(sadt_bal_accs):.1f}%
  - SEED-VIG subjects: {np.mean(seedvig_bal_accs):.1f}% ± {np.std(seedvig_bal_accs):.1f}%
  - Subject Invariance: {mean_subj_acc:.1f}% (chance = {100/n_subjects:.1f}%)

Interpretation:
  - If subject accuracy ≈ chance ({100/n_subjects:.1f}%), features are subject-invariant
  - Cross-dataset generalization shows robustness across recording conditions
""")
    
    print("Training completed!")


if __name__ == "__main__":
    main()

