#!/usr/bin/env python3
"""
Domain Adversarial Neural Network (DANN) for SADT Drowsiness Detection
========================================================================

This script implements a Domain Adversarial CNN with Gradient Reversal Layer (GRL)
for cross-subject drowsiness detection using only the SADT dataset.

The key idea: Use adversarial training to learn features that are:
  1. Discriminative for drowsiness detection (main task)
  2. Invariant to subject identity (via gradient reversal)

This forces the network to learn universal drowsiness patterns rather than
subject-specific signatures.

Architecture:
  - Feature Extractor: CNN backbone that extracts EEG features
  - Drowsiness Classifier: Predicts alert/drowsy (main task)
  - Subject Classifier: Predicts subject ID (adversarial task with GRL)

The GRL reverses gradients during backpropagation, so minimizing subject
classification loss actually maximizes confusion - making features subject-invariant.

Dataset (SADT):
  - 2022 EEG samples from 11 subjects
  - Shape: (samples, 30 channels, 384 time points) = 3s @ 128Hz
  - Labels: 0 = Alert, 1 = Drowsy

Reference:
  Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016)
  Cui et al. "A Compact and Interpretable CNN for Single-Channel EEG" (2021)

Usage:
    python train_dann_sadt.py
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
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
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
# GRADIENT REVERSAL LAYER
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for Domain Adversarial Training.
    
    Forward pass: Identity function (x -> x)
    Backward pass: Negates and scales gradients (grad -> -lambda * grad)
    
    This allows us to maximize the subject classification loss while
    minimizing the drowsiness classification loss in a single optimization step.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale by lambda
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
    Uses temporal convolutions followed by spatial filtering.
    
    Input: (batch, 1, channels=30, time=384)
    Output: (batch, feature_dim)
    """
    def __init__(self, num_channels=30, num_timepoints=384, dropout=0.5):
        super().__init__()
        
        # Temporal convolution - learns frequency filters
        # Kernel size 64 @ 128Hz captures ~0.5s patterns (good for alpha/theta rhythms)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True)
        )
        
        # Spatial convolution - learns channel combinations
        # Depthwise: each temporal filter gets its own spatial filter
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(num_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),  # Downsample temporally
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
            nn.AdaptiveAvgPool2d((1, 4)),  # Fixed output size
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
    """
    Drowsiness classifier head.
    
    Takes features from the extractor and predicts alert/drowsy.
    This is the main task we want to optimize.
    """
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
    """
    Subject classifier head (adversarial).
    
    Takes features from the extractor (after GRL) and predicts subject ID.
    The GRL ensures that optimizing this classifier actually makes
    the features LESS discriminative for subject identity.
    """
    def __init__(self, input_dim=256, hidden_dim=128, num_subjects=11, dropout=0.3):
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
    
    The network learns features that are:
      - Discriminative for drowsiness (via drowsiness classifier)
      - Invariant to subject identity (via GRL + subject classifier)
    
    Architecture:
        Input -> Feature Extractor -> [features]
                                          |
                          +---------------+---------------+
                          |                               |
                          v                               v
                  Drowsiness Classifier           GRL -> Subject Classifier
                  (minimize loss)                 (maximize confusion)
    """
    def __init__(self, num_channels=30, num_timepoints=384, 
                 num_subjects=11, dropout=0.5):
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
                       - 0: No adversarial training
                       - 1: Full adversarial training
                       
        Returns:
            drowsiness_logits: Drowsiness predictions (batch, 2)
            subject_logits: Subject predictions (batch, num_subjects)
            features: Extracted features (batch, feature_dim)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Drowsiness prediction (main task)
        drowsiness_logits = self.drowsiness_classifier(features)
        
        # Subject prediction (adversarial task)
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

class SADTDataset(Dataset):
    """
    PyTorch Dataset for SADT EEG data.
    
    Handles:
      - Z-score normalization (per-channel)
      - Adding channel dimension for Conv2d
      - Subject index conversion (1-indexed to 0-indexed)
    """
    def __init__(self, X, y, subject_ids, normalize=True, stats=None):
        """
        Args:
            X: EEG data (samples, channels, time)
            y: Drowsiness labels (0=alert, 1=drowsy)
            subject_ids: Subject IDs (1-11)
            normalize: Whether to apply z-score normalization
            stats: (mean, std) tuple for normalization. If None, compute from X.
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.subject_ids = subject_ids.astype(np.int64)
        
        # Z-score normalization per channel
        if normalize:
            if stats is None:
                # Compute stats across samples and time, keep channel dimension
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
            torch.tensor(self.subject_ids[idx] - 1)  # Convert to 0-indexed
        )
    
    def get_stats(self):
        return (self.mean, self.std)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_lambda_schedule(epoch, num_epochs, schedule='sigmoid'):
    """
    Compute lambda for gradient reversal layer.
    
    Schedules:
      - 'constant': lambda = 1.0 always
      - 'linear': lambda increases linearly from 0 to 1
      - 'sigmoid': lambda follows sigmoid curve (recommended)
    
    The sigmoid schedule allows the network to first learn good features
    for drowsiness, then gradually introduce adversarial training.
    """
    p = epoch / num_epochs
    
    if schedule == 'constant':
        return 1.0
    elif schedule == 'linear':
        return p
    elif schedule == 'sigmoid':
        # Ganin et al. schedule: 2/(1+exp(-10*p)) - 1
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs,
                alpha_subject=0.5, lambda_schedule='sigmoid'):
    """
    Train for one epoch.
    
    Args:
        model: DANN model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epoch: Current epoch
        num_epochs: Total epochs
        alpha_subject: Weight for subject classification loss
        lambda_schedule: Schedule for GRL lambda
        
    Returns:
        avg_loss: Average total loss
        avg_drowsiness_loss: Average drowsiness loss
        avg_subject_loss: Average subject loss
        drowsiness_acc: Drowsiness classification accuracy
        subject_acc: Subject classification accuracy
    """
    model.train()
    
    # Compute lambda for this epoch
    lambda_grl = compute_lambda_schedule(epoch, num_epochs, lambda_schedule)
    
    # Loss functions
    drowsiness_criterion = nn.CrossEntropyLoss()
    subject_criterion = nn.CrossEntropyLoss()
    
    # Metrics
    total_loss = 0
    total_drowsiness_loss = 0
    total_subject_loss = 0
    drowsiness_correct = 0
    subject_correct = 0
    total_samples = 0
    
    for batch_idx, (data, drowsiness_labels, subject_labels) in enumerate(train_loader):
        data = data.to(device)
        drowsiness_labels = drowsiness_labels.to(device)
        subject_labels = subject_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        drowsiness_logits, subject_logits, _ = model(data, lambda_grl)
        
        # Compute losses
        loss_drowsiness = drowsiness_criterion(drowsiness_logits, drowsiness_labels)
        loss_subject = subject_criterion(subject_logits, subject_labels)
        
        # Combined loss
        # Note: We ADD the subject loss because the GRL already reverses gradients
        # So minimizing this combined loss = minimize drowsiness loss + maximize subject confusion
        loss = loss_drowsiness + alpha_subject * loss_subject
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
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
    """
    Evaluate model on a dataset.
    
    Returns:
        loss: Average drowsiness loss
        accuracy: Drowsiness accuracy
        balanced_acc: Balanced accuracy
        subject_acc: Subject classification accuracy (should be low if DANN works)
        predictions: Drowsiness predictions
        labels: True labels
    """
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
            
            # Forward pass (lambda=0 for evaluation - no gradient reversal needed)
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

def load_sadt_data(data_path='data/SADT/sadt.mat'):
    """Load SADT dataset."""
    print("Loading SADT dataset...")
    data = loadmat(data_path)
    
    X = data['EEGsample']  # (2022, 30, 384)
    y = data['substate'].flatten()  # (2022,)
    subject_ids = data['subindex'].flatten()  # (2022,)
    
    print(f"  Shape: {X.shape} (samples, channels, time)")
    print(f"  Subjects: {len(np.unique(subject_ids))} (IDs: {np.unique(subject_ids)})")
    print(f"  Labels: Alert={np.sum(y==0)}, Drowsy={np.sum(y==1)}")
    
    return X, y, subject_ids


def main():
    print("=" * 70)
    print("Domain Adversarial CNN for SADT Drowsiness Detection")
    print("=" * 70)
    print("\nUsing Gradient Reversal Layer to learn subject-invariant features")
    
    # ==================== Configuration ====================
    data_path = 'data/SADT/sadt.mat'
    output_dir = 'results_dann_sadt'
    
    # Training hyperparameters
    num_epochs = 150
    batch_size = 32
    learning_rate = 5e-4
    weight_decay = 1e-4
    patience = 20
    
    # DANN hyperparameters
    alpha_subject = 0.5  # Weight for subject adversarial loss
    lambda_schedule = 'sigmoid'  # GRL lambda schedule
    dropout = 0.5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Load Data ====================
    X, y, subject_ids = load_sadt_data(data_path)
    
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    print(f"\nPerforming Leave-One-Subject-Out (LOSO) cross-validation")
    print(f"Testing generalization to unseen subjects\n")
    
    # ==================== Results Storage ====================
    all_results = []
    fold_accuracies = []
    fold_balanced_accs = []
    fold_subject_accs = []  # Track subject classification acc (should be low)
    
    # ==================== LOSO Cross-Validation ====================
    for fold_idx, test_subject in enumerate(unique_subjects):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1}/{n_subjects}: Test Subject = {test_subject}")
        print(f"{'='*70}")
        
        # Split data
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        subj_train = subject_ids[train_mask]
        subj_test = subject_ids[test_mask]
        
        # Create validation split (10% of training data)
        n_train = len(y_train)
        indices = np.random.permutation(n_train)
        val_size = int(0.1 * n_train)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_val, y_val = X_train[val_indices], y_train[val_indices]
        subj_val = subj_train[val_indices]
        
        X_train, y_train = X_train[train_indices], y_train[train_indices]
        subj_train = subj_train[train_indices]
        
        print(f"Train: {len(y_train)} samples (Alert: {np.sum(y_train==0)}, Drowsy: {np.sum(y_train==1)})")
        print(f"Val: {len(y_val)} samples")
        print(f"Test: {len(y_test)} samples (Alert: {np.sum(y_test==0)}, Drowsy: {np.sum(y_test==1)})")
        
        # Create datasets
        train_dataset = SADTDataset(X_train, y_train, subj_train, normalize=True)
        stats = train_dataset.get_stats()
        val_dataset = SADTDataset(X_val, y_val, subj_val, normalize=True, stats=stats)
        test_dataset = SADTDataset(X_test, y_test, subj_test, normalize=True, stats=stats)
        
        # Class-balanced sampling for training
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        # Note: num_subjects = n_subjects - 1 for training (test subject excluded)
        model = DANN(
            num_channels=30,
            num_timepoints=384,
            num_subjects=n_subjects,  # Keep full subject space
            dropout=dropout
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold_idx == 0:
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
            # Train
            train_loss, drowsiness_loss, subject_loss, train_acc, train_subj_acc, lambda_grl = train_epoch(
                model, train_loader, optimizer, device, epoch, num_epochs,
                alpha_subject=alpha_subject, lambda_schedule=lambda_schedule
            )
            
            # Validate
            val_loss, val_acc, val_bal_acc, val_subj_acc, _, _ = evaluate(model, val_loader, device)
            
            scheduler.step()
            
            # Track history
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_bal_acc'].append(val_bal_acc)
            history['subject_acc'].append(val_subj_acc)
            history['lambda'].append(lambda_grl)
            
            # Early stopping based on balanced accuracy
            if val_bal_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_bal_acc
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Logging
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Loss={train_loss:.4f} (D:{drowsiness_loss:.4f}, S:{subject_loss:.4f}) | "
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
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Balanced Accuracy: {test_bal_acc:.2f}%")
        print(f"  Subject Classifier Accuracy: {test_subj_acc:.2f}% (lower = better subject invariance)")
        
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
            'test_accuracy': test_acc,
            'balanced_accuracy': test_bal_acc
        }, os.path.join(output_dir, f'model_fold{fold_idx+1}.pt'))
    
    # ==================== Final Results ====================
    print("\n" + "=" * 70)
    print("FINAL RESULTS - Domain Adversarial CNN")
    print("=" * 70)
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_bal_acc = np.mean(fold_balanced_accs)
    std_bal_acc = np.std(fold_balanced_accs)
    mean_subj_acc = np.mean(fold_subject_accs)
    
    print(f"\nMean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Mean Balanced Accuracy: {mean_bal_acc:.2f}% ± {std_bal_acc:.2f}%")
    print(f"Mean Subject Classifier Accuracy: {mean_subj_acc:.2f}% (chance = {100/n_subjects:.1f}%)")
    
    print("\nPer-fold results:")
    print(f"{'Fold':>4} {'Subject':>7} {'Accuracy':>10} {'Balanced':>10} {'Subj Acc':>10}")
    print("-" * 45)
    for i, (acc, bal_acc, subj_acc) in enumerate(zip(fold_accuracies, fold_balanced_accs, fold_subject_accs)):
        print(f"{i+1:>4} {int(unique_subjects[i]):>7} {acc:>9.2f}% {bal_acc:>9.2f}% {subj_acc:>9.2f}%")
    
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
        'lambda_schedule': lambda_schedule
    })
    print(f"\nResults saved to {results_file}")
    
    # ==================== Plot Results ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy per fold
    ax1 = axes[0, 0]
    x = np.arange(1, n_subjects + 1)
    width = 0.35
    ax1.bar(x - width/2, fold_accuracies, width, label='Accuracy', color='steelblue')
    ax1.bar(x + width/2, fold_balanced_accs, width, label='Balanced Acc', color='coral')
    ax1.axhline(y=mean_acc, color='steelblue', linestyle='--', alpha=0.7)
    ax1.axhline(y=mean_bal_acc, color='coral', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Fold (Test Subject)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('LOSO Cross-Validation Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{int(s)}' for s in unique_subjects])
    ax1.legend()
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Subject classifier accuracy (should be low)
    ax2 = axes[0, 1]
    ax2.bar(x, fold_subject_accs, color='mediumseagreen')
    ax2.axhline(y=100/n_subjects, color='red', linestyle='--', label=f'Chance ({100/n_subjects:.1f}%)')
    ax2.axhline(y=mean_subj_acc, color='darkgreen', linestyle='--', label=f'Mean ({mean_subj_acc:.1f}%)')
    ax2.set_xlabel('Fold (Test Subject)')
    ax2.set_ylabel('Subject Classification Accuracy (%)')
    ax2.set_title('Subject Invariance (Lower = Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'S{int(s)}' for s in unique_subjects])
    ax2.legend()
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Training curves from last fold
    ax3 = axes[1, 0]
    last_history = all_results[-1]['history']
    epochs = range(1, len(last_history['train_loss']) + 1)
    ax3.plot(epochs, last_history['val_acc'], label='Val Accuracy', color='steelblue')
    ax3.plot(epochs, last_history['val_bal_acc'], label='Val Balanced Acc', color='coral')
    ax3.plot(epochs, last_history['subject_acc'], label='Subject Acc', color='mediumseagreen')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title(f'Training Curves (Fold {n_subjects})')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Lambda schedule
    ax4 = axes[1, 1]
    ax4.plot(epochs, last_history['lambda'], color='purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('λ (GRL strength)')
    ax4.set_title('Gradient Reversal Lambda Schedule')
    ax4.grid(alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dann_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, 'dann_results.png')}")
    
    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
The Domain Adversarial CNN uses gradient reversal to learn subject-invariant
features for drowsiness detection.

Key metrics:
  - Drowsiness Detection: {mean_bal_acc:.1f}% balanced accuracy
  - Subject Invariance: {mean_subj_acc:.1f}% subject accuracy (chance = {100/n_subjects:.1f}%)
  
If subject accuracy is close to chance level ({100/n_subjects:.1f}%), the model has
successfully learned features that don't encode subject-specific information.

This improves generalization to new, unseen subjects.
""")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

