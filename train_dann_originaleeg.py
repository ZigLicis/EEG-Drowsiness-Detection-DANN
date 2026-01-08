#!/usr/bin/env python3
"""
Domain Adversarial Neural Network (DANN) for TheOriginalEEG Drowsiness Detection
==================================================================================

This script trains a Domain Adversarial CNN on preprocessed EEG data from
TheOriginalEEG dataset, exported by the MATLAB preprocessing pipeline.

Data Format (from MATLAB):
  - Input: FFT spectral images (128 freq bins × 7 channels × 1)
  - Labels: Normal (1) vs Fatigued (2) → converted to 0/1
  - Subjects: 12 subjects
  - Windows: 5-second windows with 1-second stride
  - Preprocessing: 1-45 Hz bandpass, 250 Hz, ASR, ICA+ICLabel

The DANN uses gradient reversal to learn features that are:
  1. Discriminative for drowsiness detection (main task)
  2. Invariant to subject identity (adversarial task)

Usage:
    python train_dann_originaleeg.py

Requires:
    - MATLAB preprocessing to be completed first (main_dl_pipeline.m)
    - Data exported to diagnostics/python_data/fold_*.mat
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from scipy.io import savemat
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_mat_file(filepath):
    """Load .mat file, handling both v7 and v7.3 (HDF5) formats."""
    try:
        # Try scipy first (for v7 and earlier)
        from scipy.io import loadmat
        return loadmat(filepath)
    except NotImplementedError:
        # Fall back to h5py for v7.3 (HDF5) format
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if key.startswith('#'):
                    continue
                item = f[key]
                
                # Check if it's a dataset
                if isinstance(item, h5py.Dataset):
                    val = item[()]
                    
                    # Handle object references (cell arrays of strings)
                    if val.dtype == object or (hasattr(val, 'dtype') and 'ref' in str(val.dtype)):
                        try:
                            strings = []
                            for ref in val.flatten():
                                if isinstance(ref, h5py.Reference):
                                    deref = f[ref]
                                    if isinstance(deref, h5py.Dataset):
                                        chars = deref[()].flatten()
                                        s = ''.join(chr(int(c)) for c in chars)
                                        strings.append(s)
                            if strings:
                                data[key] = np.array(strings)
                                continue
                        except Exception as e:
                            pass
                    
                    # Transpose arrays to match MATLAB's column-major order
                    if isinstance(val, np.ndarray) and val.ndim > 1:
                        val = val.T
                    data[key] = val
                    
                elif isinstance(item, h5py.Group):
                    # Skip groups (like #refs#)
                    pass
                    
        return data

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for Domain Adversarial Training."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal."""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


# ============================================================================
# FEATURE EXTRACTOR (CNN for Spectral Images)
# ============================================================================

class SpectralFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor for FFT spectral images.
    
    Input: (batch, 1, freq_bins=128, channels=7)
    Output: (batch, feature_dim)
    """
    def __init__(self, num_freq_bins=128, num_channels=7, dropout=0.5):
        super().__init__()
        
        # Conv block 1: Extract frequency patterns
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 128 -> 64
            nn.Dropout2d(dropout * 0.5)
        )
        
        # Conv block 2: Cross-channel patterns
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 64 -> 32
            nn.Dropout2d(dropout * 0.5)
        )
        
        # Conv block 3: Higher-level features
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 32 -> 16
            nn.Dropout2d(dropout * 0.5)
        )
        
        # Conv block 4: Abstract features
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 1)),  # Fixed output size
            nn.Dropout2d(dropout)
        )
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_freq_bins, num_channels)
            dummy = self._forward_conv(dummy)
            self.feature_dim = dummy.numel()
        
        # Feature projection
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.output_dim = 256
    
    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
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
    """Subject classifier head - adversarial task."""
    def __init__(self, input_dim=256, hidden_dim=128, num_subjects=12, dropout=0.3):
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
    """Domain Adversarial Neural Network for Cross-Subject EEG Classification."""
    
    def __init__(self, num_freq_bins=128, num_channels=7, num_subjects=12, dropout=0.5):
        super().__init__()
        
        self.feature_extractor = SpectralFeatureExtractor(
            num_freq_bins=num_freq_bins,
            num_channels=num_channels,
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
        features = self.feature_extractor(x)
        drowsiness_logits = self.drowsiness_classifier(features)
        
        self.grl.set_lambda(lambda_grl)
        reversed_features = self.grl(features)
        subject_logits = self.subject_classifier(reversed_features)
        
        return drowsiness_logits, subject_logits, features


# ============================================================================
# DATASET
# ============================================================================

class OriginalEEGDataset(Dataset):
    """PyTorch Dataset for TheOriginalEEG preprocessed data."""
    
    def __init__(self, X, y, subject_ids, normalize=True, stats=None):
        """
        Args:
            X: Spectral data (samples, freq_bins, channels, 1) or (freq_bins, channels, 1, samples)
            y: Drowsiness labels (1=Normal, 2=Fatigued) -> converted to (0, 1)
            subject_ids: Subject IDs (1-indexed)
            normalize: Whether to apply z-score normalization
            stats: (mean, std) for normalization
        """
        # Handle MATLAB's dimension ordering (freq × channels × 1 × samples)
        if X.ndim == 4 and X.shape[2] == 1:
            # MATLAB format: (freq, channels, 1, samples) -> (samples, 1, freq, channels)
            X = np.transpose(X, (3, 2, 0, 1))
        
        self.X = X.astype(np.float32)
        
        # Convert labels: MATLAB uses 1=Normal, 2=Fatigued -> Python 0=Normal, 1=Fatigued
        self.y = (y.flatten() - 1).astype(np.int64)
        
        # Subject IDs (convert to 0-indexed)
        self.subject_ids = (subject_ids.flatten() - 1).astype(np.int64)
        
        # Z-score normalization
        if normalize:
            if stats is None:
                self.mean = self.X.mean()
                self.std = self.X.std() + 1e-8
            else:
                self.mean, self.std = stats
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean, self.std = None, None
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.subject_ids[idx])
        )
    
    def get_stats(self):
        return (self.mean, self.std)


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
    
    for data, drowsiness_labels, subject_labels in train_loader:
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
        for data, drowsiness_labels, subject_labels in loader:
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
# DATA LOADING
# ============================================================================

def load_fold_data(data_dir, fold_num):
    """Load preprocessed data for a specific fold."""
    fold_file = os.path.join(data_dir, f'fold_{fold_num}_data.mat')
    
    if not os.path.exists(fold_file):
        raise FileNotFoundError(f"Fold data not found: {fold_file}\n"
                               f"Please run the MATLAB preprocessing first (main_dl_pipeline.m)")
    
    print(f"Loading fold {fold_num} data from: {fold_file}")
    data = load_mat_file(fold_file)
    
    # Extract data
    X_train = data['XTrain']
    y_train = data['YTrain_numeric']
    subj_train = data['train_subject_nums']
    
    X_val = data['XValidation']
    y_val = data['YValidation_numeric']
    subj_val = data['val_subject_nums']
    
    X_test = data['XTest']
    y_test = data['YTest_numeric']
    subj_test = data['test_subject_nums']
    
    # Get unique subjects
    unique_subjects = data['unique_subjects']
    if isinstance(unique_subjects, np.ndarray):
        unique_subjects = [s.strip() for s in unique_subjects.flatten()]
    
    return {
        'X_train': X_train, 'y_train': y_train, 'subj_train': subj_train,
        'X_val': X_val, 'y_val': y_val, 'subj_val': subj_val,
        'X_test': X_test, 'y_test': y_test, 'subj_test': subj_test,
        'unique_subjects': unique_subjects
    }


def get_num_folds(data_dir):
    """Get the number of available folds."""
    fold_files = glob.glob(os.path.join(data_dir, 'fold_*_data.mat'))
    return len(fold_files)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 70)
    print("Domain Adversarial CNN for TheOriginalEEG Drowsiness Detection")
    print("=" * 70)
    print("\nUsing Gradient Reversal Layer to learn subject-invariant features\n")
    
    # ==================== Configuration ====================
    data_dir = 'diagnostics/python_data_no_ica'
    output_dir = 'results_dann_originaleeg'
    
    # Training hyperparameters
    num_epochs = 150
    batch_size = 64
    learning_rate = 5e-4
    weight_decay = 1e-4
    patience = 20
    
    # DANN hyperparameters
    alpha_subject = 0.5
    lambda_schedule = 'sigmoid'
    dropout = 0.5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Check Data ====================
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("Please run the MATLAB preprocessing pipeline first:")
        print("  1. Open MATLAB")
        print("  2. Run: main_dl_pipeline.m")
        print("  3. Wait for preprocessing to complete")
        print("  4. Re-run this Python script")
        return
    
    n_folds = get_num_folds(data_dir)
    if n_folds == 0:
        print(f"\nERROR: No fold data found in {data_dir}")
        print("Please run the MATLAB preprocessing pipeline first.")
        return
    
    print(f"Found {n_folds} folds of preprocessed data")
    
    # ==================== Load Metadata ====================
    metadata_file = os.path.join(data_dir, 'metadata.mat')
    if os.path.exists(metadata_file):
        metadata = load_mat_file(metadata_file)
        print(f"\nDataset Metadata:")
        print(f"  Data shape: {metadata['data_shape'].flatten()}")
        print(f"  Num classes: {metadata['num_classes'].flatten()[0]}")
        print(f"  Num subjects: {metadata['num_subjects'].flatten()[0]}")
    
    # ==================== Results Storage ====================
    all_results = []
    fold_accuracies = []
    fold_balanced_accs = []
    fold_subject_accs = []
    
    # ==================== LOSO Cross-Validation ====================
    print(f"\nPerforming LOSO cross-validation across {n_folds} folds\n")
    
    for fold_idx in range(1, n_folds + 1):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx}/{n_folds}")
        print(f"{'='*70}")
        
        # Load fold data
        try:
            fold_data = load_fold_data(data_dir, fold_idx)
        except FileNotFoundError as e:
            print(f"  Skipping fold {fold_idx}: {e}")
            continue
        
        X_train = fold_data['X_train']
        y_train = fold_data['y_train']
        subj_train = fold_data['subj_train']
        X_val = fold_data['X_val']
        y_val = fold_data['y_val']
        subj_val = fold_data['subj_val']
        X_test = fold_data['X_test']
        y_test = fold_data['y_test']
        subj_test = fold_data['subj_test']
        unique_subjects = fold_data['unique_subjects']
        
        n_subjects = len(unique_subjects)
        
        # Get data dimensions
        if X_train.ndim == 4 and X_train.shape[2] == 1:
            num_freq_bins = X_train.shape[0]
            num_channels = X_train.shape[1]
        else:
            num_freq_bins = X_train.shape[1] if X_train.ndim == 4 else 128
            num_channels = X_train.shape[2] if X_train.ndim == 4 else 7
        
        print(f"  Data shape: freq={num_freq_bins}, channels={num_channels}")
        print(f"  Train: {X_train.shape[-1] if X_train.ndim == 4 else len(y_train)} samples")
        print(f"  Val: {X_val.shape[-1] if X_val.ndim == 4 else len(y_val)} samples")
        print(f"  Test: {X_test.shape[-1] if X_test.ndim == 4 else len(y_test)} samples")
        print(f"  Subjects: {n_subjects}")
        
        # Create datasets
        train_dataset = OriginalEEGDataset(X_train, y_train, subj_train, normalize=True)
        stats = train_dataset.get_stats()
        val_dataset = OriginalEEGDataset(X_val, y_val, subj_val, normalize=True, stats=stats)
        test_dataset = OriginalEEGDataset(X_test, y_test, subj_test, normalize=True, stats=stats)
        
        # Class-balanced sampling
        y_train_np = train_dataset.y
        class_counts = np.bincount(y_train_np)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train_np]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = DANN(
            num_freq_bins=num_freq_bins,
            num_channels=num_channels,
            num_subjects=n_subjects,
            dropout=dropout
        ).to(device)
        
        # Count parameters (only on first fold)
        if fold_idx == 1:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n  Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
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
                print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Loss={train_loss:.4f} | "
                      f"Train={train_acc:.1f}% | Val Bal={val_bal_acc:.1f}% | "
                      f"Subj={val_subj_acc:.1f}% | λ={lambda_grl:.3f}")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)
        
        test_loss, test_acc, test_bal_acc, test_subj_acc, preds, labels = evaluate(model, test_loader, device)
        
        print(f"\n  Fold {fold_idx} Results:")
        print(f"    Test Accuracy: {test_acc:.2f}%")
        print(f"    Balanced Accuracy: {test_bal_acc:.2f}%")
        print(f"    Subject Classifier Accuracy: {test_subj_acc:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        print(f"    Confusion Matrix:")
        print(f"      Normal  -> Normal: {cm[0,0]:3d}, Fatigued: {cm[0,1]:3d}")
        print(f"      Fatigued -> Normal: {cm[1,0]:3d}, Fatigued: {cm[1,1]:3d}")
        
        # Store results
        fold_accuracies.append(test_acc)
        fold_balanced_accs.append(test_bal_acc)
        fold_subject_accs.append(test_subj_acc)
        
        all_results.append({
            'fold': fold_idx,
            'test_accuracy': test_acc,
            'balanced_accuracy': test_bal_acc,
            'subject_accuracy': test_subj_acc,
            'predictions': preds,
            'labels': labels,
            'history': history
        })
        
        # Save model
        torch.save({
            'model_state_dict': best_model_state,
            'fold': fold_idx,
            'test_accuracy': test_acc,
            'balanced_accuracy': test_bal_acc
        }, os.path.join(output_dir, f'model_fold{fold_idx}.pt'))
    
    # ==================== Final Results ====================
    print("\n" + "=" * 70)
    print("FINAL RESULTS - Domain Adversarial CNN (TheOriginalEEG)")
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
    
    print("\nPer-fold results:")
    print(f"{'Fold':>4} {'Accuracy':>10} {'Balanced':>10} {'Subj Acc':>10}")
    print("-" * 40)
    for i, (acc, bal_acc, subj_acc) in enumerate(zip(fold_accuracies, fold_balanced_accs, fold_subject_accs)):
        print(f"{i+1:>4} {acc:>9.2f}% {bal_acc:>9.2f}% {subj_acc:>9.2f}%")
    
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
        'alpha_subject': alpha_subject,
        'lambda_schedule': lambda_schedule
    })
    print(f"\nResults saved to {results_file}")
    
    # ==================== Plot Results ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy per fold
    ax1 = axes[0, 0]
    x = np.arange(1, len(fold_accuracies) + 1)
    width = 0.35
    ax1.bar(x - width/2, fold_accuracies, width, label='Accuracy', color='steelblue')
    ax1.bar(x + width/2, fold_balanced_accs, width, label='Balanced Acc', color='coral')
    ax1.axhline(y=mean_acc, color='steelblue', linestyle='--', alpha=0.7)
    ax1.axhline(y=mean_bal_acc, color='coral', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('LOSO Cross-Validation Results (TheOriginalEEG)')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Subject classifier accuracy
    ax2 = axes[0, 1]
    ax2.bar(x, fold_subject_accs, color='mediumseagreen')
    ax2.axhline(y=100/n_subjects, color='red', linestyle='--', label=f'Chance ({100/n_subjects:.1f}%)')
    ax2.axhline(y=mean_subj_acc, color='darkgreen', linestyle='--', label=f'Mean ({mean_subj_acc:.1f}%)')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Subject Classification Accuracy (%)')
    ax2.set_title('Subject Invariance (Lower = Better)')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.set_ylim([0, max(fold_subject_accs) * 1.2])
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Training curves from last fold
    ax3 = axes[1, 0]
    if all_results:
        last_history = all_results[-1]['history']
        epochs = range(1, len(last_history['train_loss']) + 1)
        ax3.plot(epochs, last_history['val_acc'], label='Val Accuracy', color='steelblue')
        ax3.plot(epochs, last_history['val_bal_acc'], label='Val Balanced Acc', color='coral')
        ax3.plot(epochs, last_history['subject_acc'], label='Subject Acc', color='mediumseagreen')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title(f'Training Curves (Last Fold)')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # Plot 4: Lambda schedule
    ax4 = axes[1, 1]
    if all_results:
        ax4.plot(epochs, last_history['lambda'], color='purple', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('λ (GRL strength)')
        ax4.set_title('Gradient Reversal Lambda Schedule')
        ax4.grid(alpha=0.3)
        ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dann_originaleeg_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, 'dann_originaleeg_results.png')}")
    
    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Domain Adversarial CNN trained on TheOriginalEEG dataset using gradient
reversal to learn subject-invariant drowsiness features.

Dataset:
  - Source: TheOriginalEEG (preprocessed by MATLAB pipeline)
  - Channels: 7 frontal (FP1, FP2, F7, F3, FZ, F4, F8)
  - Features: FFT spectral images ({num_freq_bins} freq bins × {num_channels} channels)
  - Subjects: {n_subjects}
  - Labels: Normal vs Fatigued

Key Results:
  - Balanced Accuracy: {mean_bal_acc:.1f}% ± {std_bal_acc:.1f}%
  - Subject Invariance: {mean_subj_acc:.1f}% (chance = {100/n_subjects:.1f}%)

Interpretation:
  - If subject accuracy ≈ chance ({100/n_subjects:.1f}%), features are subject-invariant
  - This improves generalization to new, unseen subjects
""")
    
    print("Training completed!")


if __name__ == "__main__":
    main()

