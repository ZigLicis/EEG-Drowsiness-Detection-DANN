#!/usr/bin/env python3
"""
Domain Adversarial Training for EEG Drowsiness Detection
========================================================

This script implements proper Domain Adversarial Neural Networks (DANN) for 
cross-subject EEG drowsiness detection, addressing the high variance issue
observed in MATLAB cross-validation.

Requirements:
    pip install torch torchvision scipy matplotlib scikit-learn

Usage:
    python step4_domain_adversarial_training.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Simple seed setting
torch.manual_seed(43)
np.random.seed(43)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(43)

def load_matlab_v73(filename):
    """Load MATLAB v7.3 files using h5py"""
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):  # Skip metadata
                continue
            try:
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    # Handle different data types
                    if item.dtype.char == 'U':  # Unicode strings
                        data[key] = [''.join(chr(c[0]) for c in f[item[0][0]][:].T)]
                    elif len(item.shape) == 2 and item.shape[0] == 1:
                        data[key] = item[0, 0] if item.size == 1 else item[0, :]
                    else:
                        data[key] = item[:]
                        # Transpose if needed (MATLAB vs Python array ordering)
                        if len(data[key].shape) > 2:
                            data[key] = np.transpose(data[key])
            except Exception as e:
                print(f"Warning: Could not load {key}: {e}")
                continue
    return data

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for Domain Adversarial Training
    Forward: identity function
    Backward: multiply gradients by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

class CosineClassifier(nn.Module):
    """
    Weight-normalized cosine classifier with learnable temperature (scale).
    Computes logits = s * cos(theta(f, w_c)) where f and w_c are L2-normalized.
    """
    def __init__(self, input_dim, num_classes, init_scale=16.0):
        super(CosineClassifier, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        # x: (batch, input_dim)
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        logits = self.scale * torch.matmul(x_norm, w_norm.t())
        return logits

class DomainAdversarialCNN(nn.Module):
    """
    Domain Adversarial CNN for EEG Drowsiness Detection
    
    Architecture:
    - Shared feature extractor (3 conv blocks)
    - Drowsiness classifier branch
    - Subject classifier branch (with gradient reversal)
    """
    
    def __init__(self, input_shape, num_classes, num_subjects):
        super(DomainAdversarialCNN, self).__init__()
        
        # Input shape: (batch, 1, freq_bins, channels)
        self.freq_bins, self.channels = input_shape[1], input_shape[2]
        
        # Shared feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),  # Pool only frequency
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        # Calculate feature dimensions after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.freq_bins, self.channels)
            dummy_features = self.features(dummy_input)
            self.feature_dim = dummy_features.numel()
        
        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # Normalize shared features for stability under GRL/domain shifts
        self.shared_norm = nn.LayerNorm(128)
        
        # Cosine classifier heads with learnable temperature
        self.drowsiness_classifier = CosineClassifier(128, num_classes)
        self.subject_classifier = CosineClassifier(128, num_subjects)
        
    def forward(self, x, lambda_=1.0):
        # Shared feature extraction
        features = self.features(x)
        features = features.view(features.size(0), -1)
        shared_features = self.shared_fc(features)
        shared_features = self.shared_norm(shared_features)
        
        # Drowsiness prediction (normal forward)
        drowsiness_pred = self.drowsiness_classifier(shared_features)
        
        # Subject prediction (with gradient reversal)
        reversed_features = gradient_reversal(shared_features, lambda_)
        subject_pred = self.subject_classifier(reversed_features)
        
        return drowsiness_pred, subject_pred

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, X, y_drowsiness, y_subject):
        self.X = torch.FloatTensor(X)
        self.y_drowsiness = torch.LongTensor(y_drowsiness - 1)  # Convert to 0-based
        self.y_subject = torch.LongTensor(y_subject - 1)  # Convert to 0-based
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_drowsiness[idx], self.y_subject[idx]

def train_domain_adversarial_model(train_loader, val_loader, model, device, 
                                 num_epochs=120, lr=0.0003):
    """
    Train the domain adversarial model
    """
    # Loss functions
    drowsiness_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    subject_criterion = nn.CrossEntropyLoss()
    
    # Optimizer and LR schedule: AdamW + cosine decay with warmup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_epochs = max(1, int(0.1 * num_epochs))
    cosine_epochs = max(1, num_epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr * 0.1)
    
    # Training history
    train_history = {'drowsiness_loss': [], 'subject_loss': [], 'total_loss': []}
    val_history = {'accuracy': [], 'loss': []}
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_drowsiness_loss = 0.0
        epoch_subject_loss = 0.0
        epoch_total_loss = 0.0
        
        # Warmup LR for first warmup_epochs, then cosine schedule
        if epoch < warmup_epochs:
            warmup_factor = float(epoch + 1) / float(warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * warmup_factor
        else:
            scheduler.step()

        # Dynamic lambda for gradient reversal (more aggressive schedule)
        lambda_ = 2.0 / (1.0 + np.exp(-25 * epoch / num_epochs)) - 1.0

        # Dynamic domain loss weight ramp (0.2 -> 0.8 over first 20% epochs)
        min_w, max_w, ramp_frac = 0.2, 0.8, 0.2
        progress = epoch / num_epochs
        ramp = min(1.0, progress / ramp_frac)
        domain_weight = min_w + (max_w - min_w) * ramp
        
        for batch_idx, (data, drowsiness_labels, subject_labels) in enumerate(train_loader):
            data = data.to(device)
            drowsiness_labels = drowsiness_labels.to(device)
            subject_labels = subject_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            drowsiness_pred, subject_pred = model(data, lambda_)
            
            # Calculate losses
            drowsiness_loss = drowsiness_criterion(drowsiness_pred, drowsiness_labels)
            subject_loss = subject_criterion(subject_pred, subject_labels)
            
            # Total loss (we want to minimize drowsiness loss, maximize subject loss)
            total_loss = drowsiness_loss + domain_weight * subject_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_drowsiness_loss += drowsiness_loss.item()
            epoch_subject_loss += subject_loss.item()
            epoch_total_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, drowsiness_labels, subject_labels in val_loader:
                data = data.to(device)
                drowsiness_labels = drowsiness_labels.to(device)
                
                drowsiness_pred, _ = model(data, 0.0)  # No gradient reversal in validation
                val_loss += drowsiness_criterion(drowsiness_pred, drowsiness_labels).item()
                
                _, predicted = torch.max(drowsiness_pred.data, 1)
                val_total += drowsiness_labels.size(0)
                val_correct += (predicted == drowsiness_labels).sum().item()
        
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        
        # Scheduler is stepped per-epoch above (after warmup)
        
        # Save history
        train_history['drowsiness_loss'].append(epoch_drowsiness_loss / len(train_loader))
        train_history['subject_loss'].append(epoch_subject_loss / len(train_loader))
        train_history['total_loss'].append(epoch_total_loss / len(train_loader))
        val_history['accuracy'].append(val_acc)
        val_history['loss'].append(val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Drowsiness Loss: {epoch_drowsiness_loss/len(train_loader):.4f}, '
                  f'Subject Loss: {epoch_subject_loss/len(train_loader):.4f}, '
                  f'Val Acc: {val_acc:.4f}, Lambda: {lambda_:.4f}, '
                  f'DomainW: {domain_weight:.2f}, LR: {current_lr:.6f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_history, val_history

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, drowsiness_labels, _ in test_loader:
            data = data.to(device)
            drowsiness_pred, _ = model(data, 0.0)
            
            _, predicted = torch.max(drowsiness_pred.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(drowsiness_labels.numpy())
    
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return accuracy, predictions, true_labels

def main(data_dir='diagnostics/python_data', exclude_subjects=None):
    """Main training loop for all folds
    
    Args:
        data_dir: Directory containing fold data
        exclude_subjects: List of subject IDs to exclude (e.g., [10] to exclude Subject 10)
    """
    print("=== Domain Adversarial Training for EEG Drowsiness Detection ===")
    print(f"Using data directory: {data_dir}")
    
    if exclude_subjects:
        print(f"Excluding subjects: {exclude_subjects}")
    
    # Check if data exists
    export_check_file = os.path.join(data_dir, 'export_complete.mat')
    if not os.path.exists(export_check_file):
        # Try alternate name for Cui dataset
        export_check_file = os.path.join(data_dir, 'export_complete_cui.mat')
        if not os.path.exists(export_check_file):
            print(f"Error: No exported data found in {data_dir}")
            print("Run MATLAB preprocessing first (main_dl_pipeline.m or main_dl_pipeline_cui_dataset.m)")
            return
    
    # Load metadata using h5py for v7.3 files
    metadata_file = os.path.join(data_dir, 'metadata.mat')
    try:
        metadata = load_matlab_v73(metadata_file)
        num_classes = int(metadata['num_classes'])
        num_subjects = int(metadata['num_subjects'])
        data_shape = metadata['data_shape']
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    print(f"Data shape: {data_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of subjects: {num_subjects}")
    
    # Check available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find all fold files
    fold_files = [f for f in os.listdir(data_dir) if f.startswith('fold_') and f.endswith('_data.mat')]
    fold_numbers = sorted([int(f.split('_')[1]) for f in fold_files])
    
    print(f"Found {len(fold_numbers)} folds to process")
    
    # Results storage
    fold_accuracies = []
    all_predictions = {}
    all_true_labels = {}
    
    # Process each fold
    for fold_num in fold_numbers:
        print(f"\n=== Processing Fold {fold_num} ===")
        
        # Load fold data using h5py
        try:
            fold_data = load_matlab_v73(os.path.join(data_dir, f'fold_{fold_num}_data.mat'))
        except Exception as e:
            print(f"Error loading fold {fold_num}: {e}")
            continue
        
        # Extract data
        X_train = fold_data['XTrain']
        y_train = fold_data['YTrain_numeric'].flatten()
        subject_train = fold_data['train_subject_nums'].flatten()
        
        X_val = fold_data['XValidation']
        y_val = fold_data['YValidation_numeric'].flatten()
        subject_val = fold_data['val_subject_nums'].flatten()
        
        X_test = fold_data['XTest']
        y_test = fold_data['YTest_numeric'].flatten()
        subject_test = fold_data['test_subject_nums'].flatten()
        
        # Check if test subject should be excluded
        if exclude_subjects is not None:
            test_subject_id = int(subject_test[0])  # All test samples are from same subject
            if test_subject_id in exclude_subjects:
                print(f"  Skipping Fold {fold_num} (test subject {test_subject_id} is excluded)")
                continue
            
            # Also filter excluded subjects from training and validation sets
            train_mask = ~np.isin(subject_train, exclude_subjects)
            val_mask = ~np.isin(subject_val, exclude_subjects)
            
            if train_mask.sum() == 0:
                print(f"  Skipping Fold {fold_num} (no training data after excluding subjects)")
                continue
            
            # Apply masks
            X_train = X_train[:, :, :, train_mask]
            y_train = y_train[train_mask]
            subject_train = subject_train[train_mask]
            
            if val_mask.sum() > 0:
                X_val = X_val[:, :, :, val_mask]
                y_val = y_val[val_mask]
                subject_val = subject_val[val_mask]
            
            excluded_from_train = (~train_mask).sum()
            excluded_from_val = (~val_mask).sum()
            if excluded_from_train > 0 or excluded_from_val > 0:
                print(f"  Excluded {excluded_from_train} train samples, {excluded_from_val} val samples from excluded subjects")
        
        # Reshape data for PyTorch (add channel dimension)
        X_train = np.transpose(X_train, (3, 2, 0, 1))  # (samples, channels, freq, electrodes)
        X_val = np.transpose(X_val, (3, 2, 0, 1))
        X_test = np.transpose(X_test, (3, 2, 0, 1))
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # === CHANNEL-WISE NORMALIZATION ===
        # Normalize each electrode independently to handle different amplitude ranges
        # Shape: (samples, 1, freq_bins, n_electrodes=7)
        # Compute mean/std per electrode across all samples and frequencies (from training set only)
        print("Applying per-electrode (channel-wise) normalization...")
        
        # For each electrode (axis 3), compute statistics across samples and frequencies
        train_mean_per_channel = X_train.mean(axis=(0, 1, 2), keepdims=True)  # Shape: (1, 1, 1, 7)
        train_std_per_channel = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8  # Shape: (1, 1, 1, 7)
        
        # Apply normalization (use training statistics for all sets)
        X_train = (X_train - train_mean_per_channel) / train_std_per_channel
        X_val = (X_val - train_mean_per_channel) / train_std_per_channel
        X_test = (X_test - train_mean_per_channel) / train_std_per_channel
        
        print(f"  Channel means (before): {train_mean_per_channel.squeeze()}")
        print(f"  Channel stds (before): {train_std_per_channel.squeeze()}")
        print(f"  Train data after norm - mean: {X_train.mean():.6f}, std: {X_train.std():.6f}")
        print(f"  Per-channel stats after norm - means: {X_train.mean(axis=(0,1,2)).squeeze()}")
        
        # Create datasets
        train_dataset = EEGDataset(X_train, y_train, subject_train)
        val_dataset = EEGDataset(X_val, y_val, subject_val)
        test_dataset = EEGDataset(X_test, y_test, subject_test)
        
        # Balanced domain sampling: weight samples so domains are balanced
        # Infer domain id from subject id: assume subjects 1..N_orig are original, rest are DROZY
        # We detect domain by checking unique_subjects names
        unique_subjects_py = metadata['unique_subjects'] if 'unique_subjects' in metadata else None
        domain_weights = None
        if unique_subjects_py is not None:
            # Build subject->domain map
            subs = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in unique_subjects_py]
            subj_to_domain = {i: (1 if s.startswith('Drozy_') else 0) for i, s in enumerate(subs)}
            # Convert subject labels (0-based) to domain labels
            dom_train = np.array([subj_to_domain[int(s)] for s in (subject_train - 1)])
            # Compute class weights inverse to domain freq
            counts = np.bincount(dom_train, minlength=2) + 1e-6
            weights = 1.0 / counts
            sample_weights = weights[dom_train]
            import torch.utils.data as tud
            sampler = tud.WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), 
                                                 num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        input_shape = (1, X_train.shape[2], X_train.shape[3])  # (channels, freq, electrodes)
        model = DomainAdversarialCNN(input_shape, num_classes, num_subjects).to(device)
        
        print(f"Model created with input shape: {input_shape}")
        
        # Train model
        model, train_history, val_history = train_domain_adversarial_model(
            train_loader, val_loader, model, device, num_epochs=120, lr=0.0003
        )
        
        # Evaluate on test set
        test_accuracy, predictions, true_labels = evaluate_model(model, test_loader, device)
        # Per-domain test accuracy logging (if metadata has subjects)
        per_domain_acc = {}
        if unique_subjects_py is not None:
            subs = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in unique_subjects_py]
            subj_to_domain = {i: (1 if s.startswith('Drozy_') else 0) for i, s in enumerate(subs)}
            dom_test = np.array([subj_to_domain[int(s)] for s in (subject_test - 1)])
            pred_np = np.array(predictions)
            true_np = np.array(true_labels)
            for d in [0,1]:
                mask = dom_test == d
                if mask.any():
                    per_domain_acc['orig' if d==0 else 'drozy'] = float((pred_np[mask] == true_np[mask]).mean())
        
        print(f"Fold {fold_num} Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Store results (with fold number for tracking)
        fold_accuracies.append((fold_num, test_accuracy))
        all_predictions[fold_num] = predictions
        all_true_labels[fold_num] = true_labels
        
        # Save fold results
        fold_results = {
            'fold_num': fold_num,
            'test_accuracy': test_accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'train_history': train_history,
            'val_history': val_history,
            'per_domain_acc': per_domain_acc if 'per_domain_acc' in locals() else {}
        }
        savemat(os.path.join(data_dir, f'fold_{fold_num}_results.mat'), fold_results)
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Fold {fold_num}: Accuracy {test_accuracy:.2f}%')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_save_path = os.path.join(data_dir, f'confusion_matrix_fold_{fold_num}.png')
        plt.savefig(cm_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_save_path}")
    
    # Final results
    print("\n=== FINAL RESULTS ===")
    
    # Extract just the accuracy values
    accuracies_only = [acc for _, acc in fold_accuracies]
    
    print(f"Processed {len(fold_accuracies)} folds")
    if exclude_subjects:
        print(f"Excluded subjects: {exclude_subjects}")
    
    print(f"Mean CV Accuracy: {np.mean(accuracies_only):.4f} ± {np.std(accuracies_only):.4f}")
    print(f"Mean CV Accuracy: {np.mean(accuracies_only)*100:.2f}% ± {np.std(accuracies_only)*100:.2f}%")
    
    print("\nIndividual fold accuracies:")
    for fold_num, acc in fold_accuracies:
        print(f"Fold {fold_num}: {acc*100:.2f}%")
    
    if np.std(accuracies_only) < 0.05:
        print("\nLOW VARIANCE: Results are consistent across folds (excellent!)")
    else:
        print("\nHIGH VARIANCE: Results still vary across folds (may need more regularization)")
    
    # Save final results
    final_results = {
        'fold_accuracies': accuracies_only,
        'fold_numbers': [f for f, _ in fold_accuracies],
        'mean_accuracy': np.mean(accuracies_only),
        'std_accuracy': np.std(accuracies_only),
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels,
        'excluded_subjects': exclude_subjects if exclude_subjects else []
    }
    final_results_path = os.path.join(data_dir, 'final_results.mat')
    savemat(final_results_path, final_results)
    
    print(f"\nResults saved to {final_results_path}")
    print("Domain adversarial training completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Domain Adversarial Training for EEG Drowsiness Detection')
    parser.add_argument('--data_dir', type=str, default='diagnostics/python_data',
                        help='Directory containing exported MATLAB data (default: diagnostics/python_data)')
    parser.add_argument('--exclude_subjects', type=int, nargs='*', default=None,
                        help='Subject IDs to exclude from training/testing (e.g., --exclude_subjects 10 11)')
    args = parser.parse_args()
    
    main(data_dir=args.data_dir, exclude_subjects=args.exclude_subjects) 