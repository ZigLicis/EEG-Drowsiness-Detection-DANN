#!/usr/bin/env python3
"""
Ablation Study: Comparing DANN vs Baseline Models
==================================================

This script implements baseline models (SVM, CNN, CNN-LSTM) for comparison
against the Domain Adversarial Neural Network (DANN) approach.

Models:
    1. SVM - Support Vector Machine with RBF kernel on flattened spectral features
    2. CNN - Same architecture as DANN but without domain adversarial training
    3. CNN-LSTM - CNN feature extractor + LSTM for temporal modeling

Usage:
    python ablation_study.py --data_dir python_data_best\ 13-04-03-643 --model all
    python ablation_study.py --data_dir python_data_best\ 13-04-03-643 --model svm
    python ablation_study.py --data_dir python_data_best\ 13-04-03-643 --model cnn
    python ablation_study.py --data_dir python_data_best\ 13-04-03-643 --model cnn_lstm
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
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
            except Exception as e:
                print(f"Warning: Could not load {key}: {e}")
                continue
    return data

# ============================================================================
# Dataset Classes
# ============================================================================

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y - 1)  # Convert to 0-based
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# Model 1: SVM (Support Vector Machine)
# ============================================================================

class SVMClassifier:
    """SVM classifier for EEG drowsiness detection"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.scaler = StandardScaler()
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=43)
        
    def fit(self, X_train, y_train):
        # Flatten the spectral features
        X_flat = X_train.reshape(X_train.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y_train)
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# ============================================================================
# Model 2: CNN (without Domain Adversarial Training)
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN for EEG classification (same architecture as DANN feature extractor)
    but without the domain adversarial branch
    """
    
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.freq_bins, self.channels = input_shape[1], input_shape[2]
        
        # Feature extractor (same as DANN)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
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
        
        # Calculate feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.freq_bins, self.channels)
            dummy_features = self.features(dummy_input)
            self.feature_dim = dummy_features.numel()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

# ============================================================================
# Model 3: CNN-LSTM
# ============================================================================

class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for EEG classification
    Uses CNN for spatial feature extraction and LSTM for temporal modeling
    """
    
    def __init__(self, input_shape, num_classes, hidden_size=64, num_layers=2):
        super(CNNLSTM, self).__init__()
        
        self.freq_bins, self.channels = input_shape[1], input_shape[2]
        
        # CNN feature extractor (processes each frequency band)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        # Calculate CNN output dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.freq_bins, self.channels)
            cnn_out = self.cnn(dummy_input)
            self.cnn_out_freq = cnn_out.shape[2]
            self.cnn_out_channels = cnn_out.shape[1] * cnn_out.shape[3]
        
        # LSTM for temporal modeling (treat frequency as sequence)
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch, channels, freq, electrodes)
        
        # Reshape for LSTM: (batch, seq_len, features)
        batch_size = cnn_features.size(0)
        # Treat frequency dimension as sequence
        cnn_features = cnn_features.permute(0, 2, 1, 3)  # (batch, freq, channels, electrodes)
        cnn_features = cnn_features.reshape(batch_size, self.cnn_out_freq, -1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Use last hidden state from both directions
        h_forward = h_n[-2, :, :]  # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        # Classification
        output = self.classifier(h_combined)
        return output

# ============================================================================
# Training Functions
# ============================================================================

def train_pytorch_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, model_name='model'):
    """Train a PyTorch model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

def evaluate_pytorch_model(model, test_loader, device):
    """Evaluate a PyTorch model"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels

# ============================================================================
# Main Ablation Study
# ============================================================================

def run_ablation_study(data_dir, models_to_run=['svm', 'cnn', 'cnn_lstm']):
    """Run ablation study with specified models"""
    
    print("=" * 70)
    print("ABLATION STUDY: Comparing Baseline Models vs DANN")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Models to evaluate: {models_to_run}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check if data exists
    export_check_file = os.path.join(data_dir, 'export_complete.mat')
    if not os.path.exists(export_check_file):
        print(f"Error: No exported data found in {data_dir}")
        return
    
    # Load metadata
    metadata_file = os.path.join(data_dir, 'metadata.mat')
    try:
        metadata = load_matlab_v73(metadata_file)
        num_classes = int(metadata['num_classes'])
        num_subjects = int(metadata['num_subjects'])
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of subjects: {num_subjects}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find all fold files
    fold_files = [f for f in os.listdir(data_dir) if f.startswith('fold_') and f.endswith('_data.mat')]
    fold_numbers = sorted([int(f.split('_')[1]) for f in fold_files])
    print(f"Found {len(fold_numbers)} folds")
    
    # Results storage
    results = {model: {'accuracies': [], 'fold_nums': []} for model in models_to_run}
    
    # Process each fold
    for fold_num in fold_numbers:
        print(f"\n{'='*50}")
        print(f"Processing Fold {fold_num}")
        print(f"{'='*50}")
        
        # Load fold data
        try:
            fold_data = load_matlab_v73(os.path.join(data_dir, f'fold_{fold_num}_data.mat'))
        except Exception as e:
            print(f"Error loading fold {fold_num}: {e}")
            continue
        
        # Extract data
        X_train = fold_data['XTrain']
        y_train = fold_data['YTrain_numeric'].flatten()
        
        X_val = fold_data['XValidation']
        y_val = fold_data['YValidation_numeric'].flatten()
        
        X_test = fold_data['XTest']
        y_test = fold_data['YTest_numeric'].flatten()
        
        # Reshape for PyTorch models
        X_train_pt = np.transpose(X_train, (3, 2, 0, 1))
        X_val_pt = np.transpose(X_val, (3, 2, 0, 1))
        X_test_pt = np.transpose(X_test, (3, 2, 0, 1))
        
        # Normalize
        train_mean = X_train_pt.mean(axis=(0, 1, 2), keepdims=True)
        train_std = X_train_pt.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        
        X_train_pt = (X_train_pt - train_mean) / train_std
        X_val_pt = (X_val_pt - train_mean) / train_std
        X_test_pt = (X_test_pt - train_mean) / train_std
        
        print(f"Train: {X_train_pt.shape}, Val: {X_val_pt.shape}, Test: {X_test_pt.shape}")
        
        # =====================================================================
        # Model 1: SVM
        # =====================================================================
        if 'svm' in models_to_run:
            print("\n--- Training SVM ---")
            svm = SVMClassifier(kernel='rbf', C=10.0, gamma='scale')
            
            # Flatten for SVM
            X_train_flat = X_train_pt.reshape(X_train_pt.shape[0], -1)
            X_test_flat = X_test_pt.reshape(X_test_pt.shape[0], -1)
            
            svm.fit(X_train_flat, y_train)
            svm_predictions = svm.predict(X_test_flat)
            svm_acc = accuracy_score(y_test, svm_predictions)
            
            results['svm']['accuracies'].append(svm_acc)
            results['svm']['fold_nums'].append(fold_num)
            print(f"SVM Accuracy: {svm_acc*100:.2f}%")
        
        # =====================================================================
        # Model 2: CNN (without domain adversarial)
        # =====================================================================
        if 'cnn' in models_to_run:
            print("\n--- Training CNN ---")
            
            train_dataset = EEGDataset(X_train_pt, y_train)
            val_dataset = EEGDataset(X_val_pt, y_val)
            test_dataset = EEGDataset(X_test_pt, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            input_shape = (1, X_train_pt.shape[2], X_train_pt.shape[3])
            cnn_model = SimpleCNN(input_shape, num_classes).to(device)
            
            cnn_model, _ = train_pytorch_model(
                cnn_model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, model_name='CNN'
            )
            
            cnn_acc, cnn_preds, cnn_true = evaluate_pytorch_model(cnn_model, test_loader, device)
            results['cnn']['accuracies'].append(cnn_acc)
            results['cnn']['fold_nums'].append(fold_num)
            print(f"CNN Accuracy: {cnn_acc*100:.2f}%")
        
        # =====================================================================
        # Model 3: CNN-LSTM
        # =====================================================================
        if 'cnn_lstm' in models_to_run:
            print("\n--- Training CNN-LSTM ---")
            
            train_dataset = EEGDataset(X_train_pt, y_train)
            val_dataset = EEGDataset(X_val_pt, y_val)
            test_dataset = EEGDataset(X_test_pt, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            input_shape = (1, X_train_pt.shape[2], X_train_pt.shape[3])
            lstm_model = CNNLSTM(input_shape, num_classes, hidden_size=64, num_layers=2).to(device)
            
            lstm_model, _ = train_pytorch_model(
                lstm_model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, model_name='CNN-LSTM'
            )
            
            lstm_acc, lstm_preds, lstm_true = evaluate_pytorch_model(lstm_model, test_loader, device)
            results['cnn_lstm']['accuracies'].append(lstm_acc)
            results['cnn_lstm']['fold_nums'].append(fold_num)
            print(f"CNN-LSTM Accuracy: {lstm_acc*100:.2f}%")
    
    # =========================================================================
    # Final Results Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 70)
    
    # Create results table
    print("\nPer-Fold Accuracies (%):")
    print("-" * 70)
    header = "Fold\t"
    for model in models_to_run:
        header += f"{model.upper()}\t"
    print(header)
    print("-" * 70)
    
    for i, fold_num in enumerate(fold_numbers):
        row = f"{fold_num}\t"
        for model in models_to_run:
            if i < len(results[model]['accuracies']):
                row += f"{results[model]['accuracies'][i]*100:.1f}%\t"
            else:
                row += "N/A\t"
        print(row)
    
    print("-" * 70)
    print("\nOverall Performance:")
    print("-" * 70)
    
    summary_data = {}
    for model in models_to_run:
        accs = results[model]['accuracies']
        if len(accs) > 0:
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            summary_data[model] = {'mean': mean_acc, 'std': std_acc}
            print(f"{model.upper():10s}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    print("-" * 70)
    
    # Save results
    output_file = os.path.join(data_dir, 'ablation_results.mat')
    save_data = {
        'models': models_to_run,
        'fold_numbers': fold_numbers,
    }
    for model in models_to_run:
        save_data[f'{model}_accuracies'] = results[model]['accuracies']
        if model in summary_data:
            save_data[f'{model}_mean'] = summary_data[model]['mean']
            save_data[f'{model}_std'] = summary_data[model]['std']
    
    savemat(output_file, save_data)
    print(f"\nResults saved to: {output_file}")
    
    # Create comparison plot
    create_comparison_plot(results, models_to_run, fold_numbers, data_dir, summary_data)
    
    return results, summary_data

def create_comparison_plot(results, models_to_run, fold_numbers, data_dir, summary_data):
    """Create visualization comparing all models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Per-fold accuracies
    ax1 = axes[0]
    x = np.arange(len(fold_numbers))
    width = 0.25
    colors = {'svm': '#e74c3c', 'cnn': '#3498db', 'cnn_lstm': '#2ecc71'}
    
    for i, model in enumerate(models_to_run):
        accs = [a * 100 for a in results[model]['accuracies']]
        offset = (i - len(models_to_run)/2 + 0.5) * width
        bars = ax1.bar(x + offset, accs, width, label=model.upper(), 
                       color=colors.get(model, 'gray'), edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Fold Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_numbers)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 105])
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Overall comparison (bar chart with error bars)
    ax2 = axes[1]
    model_names = [m.upper() for m in models_to_run]
    means = [summary_data[m]['mean'] for m in models_to_run]
    stds = [summary_data[m]['std'] for m in models_to_run]
    
    bars = ax2.bar(model_names, means, yerr=stds, capsize=5,
                   color=[colors.get(m, 'gray') for m in models_to_run],
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax2.annotate(f'{mean:.1f}%\n±{std:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Model Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(data_dir, 'ablation_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ablation Study: Compare baseline models vs DANN')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing exported MATLAB data')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'svm', 'cnn', 'cnn_lstm'],
                        help='Which model(s) to run (default: all)')
    args = parser.parse_args()
    
    if args.model == 'all':
        models = ['svm', 'cnn', 'cnn_lstm']
    else:
        models = [args.model]
    
    run_ablation_study(args.data_dir, models)

