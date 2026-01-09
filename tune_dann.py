#!/usr/bin/env python3
"""
DANN Tuning Script
==================

Test different DANN configurations to find optimal hyperparameters.
Runs only DANN (not full ablation) for faster iteration.

Usage:
    python tune_dann.py --data_dir diagnostics/python_data_3 --config conservative
    python tune_dann.py --data_dir diagnostics/python_data_3 --config moderate
    python tune_dann.py --data_dir diagnostics/python_data_3 --config minimal
    python tune_dann.py --data_dir diagnostics/python_data_3 --config no_domain
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.io import savemat
import h5py
from sklearn.metrics import accuracy_score
import argparse
from datetime import datetime

torch.manual_seed(43)
np.random.seed(43)

def load_matlab_v73(filename):
    """Load MATLAB v7.3 files using h5py"""
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):
                continue
            try:
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    if item.dtype.char == 'U':
                        data[key] = [''.join(chr(c[0]) for c in f[item[0][0]][:].T)]
                    elif len(item.shape) == 2 and item.shape[0] == 1:
                        data[key] = item[0, 0] if item.size == 1 else item[0, :]
                    else:
                        data[key] = item[:]
                        if len(data[key].shape) > 2:
                            data[key] = np.transpose(data[key])
            except:
                continue
    return data

class EEGDatasetWithSubject(Dataset):
    def __init__(self, X, y_drowsiness, y_subject):
        self.X = torch.FloatTensor(X)
        self.y_drowsiness = torch.LongTensor(y_drowsiness - 1)
        self.y_subject = torch.LongTensor(y_subject - 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_drowsiness[idx], self.y_subject[idx]

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

class CosineClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, init_scale=16.0):
        super(CosineClassifier, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * torch.matmul(x_norm, w_norm.t())

class DomainAdversarialCNN(nn.Module):
    def __init__(self, input_shape, num_classes, num_subjects):
        super(DomainAdversarialCNN, self).__init__()
        
        self.freq_bins, self.channels = input_shape[1], input_shape[2]
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.freq_bins, self.channels)
            dummy_features = self.features(dummy_input)
            self.feature_dim = dummy_features.numel()
        
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.shared_norm = nn.LayerNorm(128)
        
        self.drowsiness_classifier = CosineClassifier(128, num_classes)
        self.subject_classifier = CosineClassifier(128, num_subjects)
        
    def forward(self, x, lambda_=1.0):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        shared_features = self.shared_fc(features)
        shared_features = self.shared_norm(shared_features)
        
        drowsiness_pred = self.drowsiness_classifier(shared_features)
        reversed_features = gradient_reversal(shared_features, lambda_)
        subject_pred = self.subject_classifier(reversed_features)
        
        return drowsiness_pred, subject_pred

# ============================================================================
# Different DANN Configurations
# ============================================================================

CONFIGS = {
    'conservative': {
        'description': 'Very gentle domain adaptation - prioritize task performance',
        'num_epochs': 150,
        'lr': 0.0005,
        'lambda_max': 0.3,
        'lambda_speed': 3,  # Slower ramp
        'domain_weight_min': 0.02,
        'domain_weight_max': 0.1,
        'domain_ramp_frac': 0.6,
        'patience': 30,
        'label_smoothing': 0.05,
    },
    'moderate': {
        'description': 'Balanced domain adaptation',
        'num_epochs': 120,
        'lr': 0.0003,
        'lambda_max': 0.5,
        'lambda_speed': 5,
        'domain_weight_min': 0.05,
        'domain_weight_max': 0.2,
        'domain_ramp_frac': 0.4,
        'patience': 25,
        'label_smoothing': 0.05,
    },
    'minimal': {
        'description': 'Minimal domain adaptation - almost like CNN',
        'num_epochs': 100,
        'lr': 0.001,
        'lambda_max': 0.1,
        'lambda_speed': 2,
        'domain_weight_min': 0.01,
        'domain_weight_max': 0.05,
        'domain_ramp_frac': 0.8,
        'patience': 20,
        'label_smoothing': 0.0,
    },
    'no_domain': {
        'description': 'No domain adaptation at all (lambda=0, weight=0)',
        'num_epochs': 100,
        'lr': 0.001,
        'lambda_max': 0.0,
        'lambda_speed': 1,
        'domain_weight_min': 0.0,
        'domain_weight_max': 0.0,
        'domain_ramp_frac': 1.0,
        'patience': 20,
        'label_smoothing': 0.0,
    },
}

def train_dann_with_config(model, train_loader, val_loader, device, config):
    """Train DANN with specified configuration"""
    
    cfg = CONFIGS[config]
    
    drowsiness_criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    subject_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=5e-5)
    num_epochs = cfg['num_epochs']
    warmup_epochs = max(1, int(0.1 * num_epochs))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=cfg['lr'] * 0.05
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg['lr'] * (epoch + 1) / warmup_epochs
        else:
            scheduler.step()
        
        progress = epoch / num_epochs
        
        # Lambda schedule
        lambda_ = cfg['lambda_max'] * (2.0 / (1.0 + np.exp(-cfg['lambda_speed'] * progress)) - 1.0)
        
        # Domain weight schedule
        ramp = min(1.0, progress / cfg['domain_ramp_frac'])
        domain_weight = cfg['domain_weight_min'] + (cfg['domain_weight_max'] - cfg['domain_weight_min']) * ramp
        
        for data, drowsiness_labels, subject_labels in train_loader:
            data = data.to(device)
            drowsiness_labels = drowsiness_labels.to(device)
            subject_labels = subject_labels.to(device)
            
            optimizer.zero_grad()
            drowsiness_pred, subject_pred = model(data, lambda_)
            
            drowsiness_loss = drowsiness_criterion(drowsiness_pred, drowsiness_labels)
            subject_loss = subject_criterion(subject_pred, subject_labels)
            total_loss = drowsiness_loss + domain_weight * subject_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, drowsiness_labels, _ in val_loader:
                data = data.to(device)
                drowsiness_labels = drowsiness_labels.to(device)
                drowsiness_pred, _ = model(data, 0.0)
                _, predicted = torch.max(drowsiness_pred.data, 1)
                val_total += drowsiness_labels.size(0)
                val_correct += (predicted == drowsiness_labels).sum().item()
        
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Val: {val_acc:.4f}, λ: {lambda_:.3f}, w: {domain_weight:.3f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, test_loader, device):
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
    
    return accuracy_score(true_labels, predictions)

def run_dann_tuning(data_dir, config='conservative'):
    """Run DANN with specified configuration across all folds"""
    
    print("=" * 70)
    print(f"DANN TUNING - Config: {config.upper()}")
    print("=" * 70)
    print(f"Configuration: {CONFIGS[config]['description']}")
    print(f"Data directory: {data_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load metadata
    metadata = load_matlab_v73(os.path.join(data_dir, 'metadata.mat'))
    num_classes = int(metadata['num_classes'])
    num_subjects = int(metadata['num_subjects'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find folds
    fold_files = [f for f in os.listdir(data_dir) if f.startswith('fold_') and f.endswith('_data.mat')]
    fold_numbers = sorted([int(f.split('_')[1]) for f in fold_files])
    print(f"Found {len(fold_numbers)} folds\n")
    
    accuracies = []
    
    for fold_num in fold_numbers:
        print(f"--- Fold {fold_num} ---")
        
        fold_data = load_matlab_v73(os.path.join(data_dir, f'fold_{fold_num}_data.mat'))
        
        X_train = np.transpose(fold_data['XTrain'], (3, 2, 0, 1))
        y_train = fold_data['YTrain_numeric'].flatten()
        subject_train = fold_data['train_subject_nums'].flatten()
        
        X_val = np.transpose(fold_data['XValidation'], (3, 2, 0, 1))
        y_val = fold_data['YValidation_numeric'].flatten()
        subject_val = fold_data['val_subject_nums'].flatten()
        
        X_test = np.transpose(fold_data['XTest'], (3, 2, 0, 1))
        y_test = fold_data['YTest_numeric'].flatten()
        subject_test = fold_data['test_subject_nums'].flatten()
        
        # Normalize
        train_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        train_std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        
        # Datasets
        train_dataset = EEGDatasetWithSubject(X_train, y_train, subject_train)
        val_dataset = EEGDatasetWithSubject(X_val, y_val, subject_val)
        test_dataset = EEGDatasetWithSubject(X_test, y_test, subject_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Model
        input_shape = (1, X_train.shape[2], X_train.shape[3])
        model = DomainAdversarialCNN(input_shape, num_classes, num_subjects).to(device)
        
        # Train
        model = train_dann_with_config(model, train_loader, val_loader, device, config)
        
        # Evaluate
        acc = evaluate_model(model, test_loader, device)
        accuracies.append(acc)
        print(f"  Test Accuracy: {acc*100:.2f}%\n")
    
    # Summary
    print("=" * 70)
    print(f"RESULTS - Config: {config.upper()}")
    print("=" * 70)
    print(f"\nPer-fold accuracies:")
    for i, (fold_num, acc) in enumerate(zip(fold_numbers, accuracies)):
        print(f"  Fold {fold_num}: {acc*100:.2f}%")
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    print(f"\nOverall: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("=" * 70)
    
    # Save results
    results = {
        'config': config,
        'fold_numbers': fold_numbers,
        'accuracies': accuracies,
        'mean': mean_acc,
        'std': std_acc,
    }
    output_path = os.path.join(data_dir, f'dann_tuning_{config}.mat')
    savemat(output_path, results)
    print(f"Results saved to: {output_path}")
    
    return mean_acc, std_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune DANN hyperparameters')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing exported MATLAB data')
    parser.add_argument('--config', type=str, default='conservative',
                        choices=['conservative', 'moderate', 'minimal', 'no_domain'],
                        help='DANN configuration to use')
    parser.add_argument('--all', action='store_true',
                        help='Run all configurations and compare')
    args = parser.parse_args()
    
    if args.all:
        print("\n" + "=" * 70)
        print("RUNNING ALL CONFIGURATIONS")
        print("=" * 70 + "\n")
        
        results = {}
        for config in ['no_domain', 'minimal', 'conservative', 'moderate']:
            mean, std = run_dann_tuning(args.data_dir, config)
            results[config] = (mean, std)
            print("\n")
        
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"\n{'Config':<15} {'Mean':<12} {'Std':<12} {'Description'}")
        print("-" * 70)
        for config, (mean, std) in results.items():
            print(f"{config:<15} {mean:.2f}%       ±{std:.2f}%       {CONFIGS[config]['description']}")
    else:
        run_dann_tuning(args.data_dir, args.config)

