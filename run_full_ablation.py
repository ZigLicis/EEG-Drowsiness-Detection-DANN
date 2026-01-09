#!/usr/bin/env python3
"""
Full Ablation Study Runner
==========================

This script runs the complete ablation study including:
1. SVM baseline
2. CNN baseline (no domain adversarial)
3. CNN-LSTM baseline
4. DANN (your main model)

And generates a comprehensive comparison report.

Usage:
    python run_full_ablation.py --data_dir "python_data_best 13-04-03-643"
"""

import os
import sys
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Import the ablation study module
from ablation_study import run_ablation_study, load_matlab_v73

def load_dann_results(data_dir):
    """Load DANN results from final_results.mat"""
    results_file = os.path.join(data_dir, 'final_results.mat')
    
    if not os.path.exists(results_file):
        print(f"Warning: DANN results not found at {results_file}")
        return None
    
    try:
        # Try regular load first
        data = loadmat(results_file)
        fold_accuracies = data['fold_accuracies'].flatten()
        return fold_accuracies
    except:
        try:
            # Try h5py for v7.3 files
            data = load_matlab_v73(results_file)
            fold_accuracies = np.array(data['fold_accuracies']).flatten()
            return fold_accuracies
        except Exception as e:
            print(f"Error loading DANN results: {e}")
            return None

def generate_comparison_report(ablation_results, dann_results, data_dir, fold_numbers):
    """Generate a comprehensive comparison report"""
    
    report_path = os.path.join(data_dir, 'ablation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY REPORT: DANN vs Baseline Models\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Directory: {data_dir}\n")
        f.write(f"Number of Folds: {len(fold_numbers)}\n\n")
        
        # Model descriptions
        f.write("-" * 80 + "\n")
        f.write("MODEL DESCRIPTIONS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("1. SVM (Support Vector Machine)\n")
        f.write("   - RBF kernel with C=10.0\n")
        f.write("   - Features: Flattened spectral power (freq × channels)\n")
        f.write("   - No temporal or domain adaptation\n\n")
        
        f.write("2. CNN (Convolutional Neural Network)\n")
        f.write("   - 3-block CNN (same architecture as DANN feature extractor)\n")
        f.write("   - Standard cross-entropy training\n")
        f.write("   - No domain adversarial training\n\n")
        
        f.write("3. CNN-LSTM (CNN + Long Short-Term Memory)\n")
        f.write("   - 2-block CNN for spatial features\n")
        f.write("   - Bidirectional LSTM (64 hidden units, 2 layers)\n")
        f.write("   - Treats frequency dimension as temporal sequence\n\n")
        
        f.write("4. DANN (Domain Adversarial Neural Network) [PROPOSED]\n")
        f.write("   - 3-block CNN feature extractor\n")
        f.write("   - Gradient Reversal Layer for domain adaptation\n")
        f.write("   - Subject classifier branch (adversarial)\n")
        f.write("   - Cosine classifier with learnable temperature\n")
        f.write("   - AdamW optimizer with cosine annealing\n\n")
        
        # Results table
        f.write("-" * 80 + "\n")
        f.write("PER-FOLD ACCURACY RESULTS (%)\n")
        f.write("-" * 80 + "\n\n")
        
        # Header
        f.write(f"{'Fold':<8}")
        models = ['svm', 'cnn', 'cnn_lstm']
        for model in models:
            f.write(f"{model.upper():<12}")
        f.write(f"{'DANN':<12}\n")
        f.write("-" * 56 + "\n")
        
        # Data rows
        for i, fold_num in enumerate(fold_numbers):
            f.write(f"{fold_num:<8}")
            for model in models:
                if model in ablation_results and i < len(ablation_results[model]['accuracies']):
                    acc = ablation_results[model]['accuracies'][i] * 100
                    f.write(f"{acc:<12.1f}")
                else:
                    f.write(f"{'N/A':<12}")
            
            if dann_results is not None and i < len(dann_results):
                dann_acc = dann_results[i] * 100 if dann_results[i] <= 1 else dann_results[i]
                f.write(f"{dann_acc:<12.1f}")
            else:
                f.write(f"{'N/A':<12}")
            f.write("\n")
        
        f.write("-" * 56 + "\n\n")
        
        # Summary statistics
        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"{'Model':<15}{'Mean (%)':<15}{'Std (%)':<15}{'Min (%)':<15}{'Max (%)':<15}\n")
        f.write("-" * 75 + "\n")
        
        all_results = {}
        
        for model in models:
            if model in ablation_results and len(ablation_results[model]['accuracies']) > 0:
                accs = np.array(ablation_results[model]['accuracies']) * 100
                all_results[model] = accs
                f.write(f"{model.upper():<15}{np.mean(accs):<15.2f}{np.std(accs):<15.2f}"
                       f"{np.min(accs):<15.2f}{np.max(accs):<15.2f}\n")
        
        if dann_results is not None:
            dann_accs = dann_results * 100 if dann_results.max() <= 1 else dann_results
            all_results['dann'] = dann_accs
            f.write(f"{'DANN':<15}{np.mean(dann_accs):<15.2f}{np.std(dann_accs):<15.2f}"
                   f"{np.min(dann_accs):<15.2f}{np.max(dann_accs):<15.2f}\n")
        
        f.write("-" * 75 + "\n\n")
        
        # Improvement analysis
        f.write("-" * 80 + "\n")
        f.write("DANN IMPROVEMENT OVER BASELINES\n")
        f.write("-" * 80 + "\n\n")
        
        if 'dann' in all_results:
            dann_mean = np.mean(all_results['dann'])
            
            for model in models:
                if model in all_results:
                    baseline_mean = np.mean(all_results[model])
                    improvement = dann_mean - baseline_mean
                    rel_improvement = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
                    
                    f.write(f"DANN vs {model.upper()}:\n")
                    f.write(f"  Absolute improvement: {improvement:+.2f}%\n")
                    f.write(f"  Relative improvement: {rel_improvement:+.2f}%\n\n")
        
        # Statistical significance note
        f.write("-" * 80 + "\n")
        f.write("NOTES\n")
        f.write("-" * 80 + "\n\n")
        f.write("- All models use Leave-One-Subject-Out (LOSO) cross-validation\n")
        f.write("- DANN uses domain adversarial training to learn subject-invariant features\n")
        f.write("- High variance across folds is expected due to inter-subject variability\n")
        f.write("- For statistical significance, consider paired t-test or Wilcoxon signed-rank test\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Report saved to: {report_path}")
    return all_results

def create_final_comparison_figure(all_results, data_dir, fold_numbers):
    """Create publication-quality comparison figure"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    models = list(all_results.keys())
    colors = {'svm': '#e74c3c', 'cnn': '#3498db', 'cnn_lstm': '#9b59b6', 'dann': '#27ae60'}
    labels = {'svm': 'SVM', 'cnn': 'CNN', 'cnn_lstm': 'CNN-LSTM', 'dann': 'DANN (Ours)'}
    
    # Plot 1: Box plot comparison
    ax1 = axes[0]
    data_to_plot = [all_results[m] for m in models]
    bp = ax1.boxplot(data_to_plot, labels=[labels[m] for m in models], patch_artist=True)
    
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Accuracy Distribution', fontsize=14, fontweight='bold')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([40, 105])
    
    # Plot 2: Bar chart with error bars
    ax2 = axes[1]
    means = [np.mean(all_results[m]) for m in models]
    stds = [np.std(all_results[m]) for m in models]
    x = np.arange(len(models))
    
    bars = ax2.bar(x, means, yerr=stds, capsize=5,
                   color=[colors[m] for m in models],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax2.annotate(f'{mean:.1f}±{std:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 1),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[m] for m in models])
    ax2.set_ylabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Mean ± Std Comparison', fontsize=14, fontweight='bold')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([40, 105])
    
    # Plot 3: Per-fold line plot
    ax3 = axes[2]
    x_folds = np.arange(len(fold_numbers))
    
    for model in models:
        if len(all_results[model]) == len(fold_numbers):
            ax3.plot(x_folds, all_results[model], 'o-', 
                    color=colors[model], label=labels[model], 
                    linewidth=2, markersize=6, alpha=0.8)
    
    ax3.set_xticks(x_folds)
    ax3.set_xticklabels(fold_numbers)
    ax3.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Per-Fold Performance', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(alpha=0.3)
    ax3.set_ylim([40, 105])
    
    plt.tight_layout()
    
    # Save
    fig_path = os.path.join(data_dir, 'ablation_final_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Final comparison figure saved to: {fig_path}")

def main():
    parser = argparse.ArgumentParser(description='Run full ablation study with DANN comparison')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing exported MATLAB data and DANN results')
    parser.add_argument('--skip_baselines', action='store_true',
                        help='Skip running baselines (use existing ablation_results.mat)')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    print("=" * 70)
    print("FULL ABLATION STUDY WITH DANN COMPARISON")
    print("=" * 70)
    
    # Load DANN results first
    print("\nLoading DANN results...")
    dann_results = load_dann_results(data_dir)
    
    if dann_results is not None:
        dann_accs = dann_results * 100 if dann_results.max() <= 1 else dann_results
        print(f"DANN Mean: {np.mean(dann_accs):.2f}% ± {np.std(dann_accs):.2f}%")
    else:
        print("Warning: DANN results not found. Run step4_domain_adversarial_training.py first.")
    
    # Run baseline models
    if args.skip_baselines:
        print("\nSkipping baseline training, loading existing results...")
        ablation_file = os.path.join(data_dir, 'ablation_results.mat')
        if os.path.exists(ablation_file):
            ablation_data = loadmat(ablation_file)
            ablation_results = {}
            for model in ['svm', 'cnn', 'cnn_lstm']:
                key = f'{model}_accuracies'
                if key in ablation_data:
                    ablation_results[model] = {'accuracies': ablation_data[key].flatten()}
        else:
            print("No existing ablation results found. Running baselines...")
            ablation_results, _ = run_ablation_study(data_dir, ['svm', 'cnn', 'cnn_lstm'])
    else:
        print("\nRunning baseline models (SVM, CNN, CNN-LSTM)...")
        ablation_results, _ = run_ablation_study(data_dir, ['svm', 'cnn', 'cnn_lstm'])
    
    # Get fold numbers
    fold_files = [f for f in os.listdir(data_dir) if f.startswith('fold_') and f.endswith('_data.mat')]
    fold_numbers = sorted([int(f.split('_')[1]) for f in fold_files])
    
    # Generate comparison report
    print("\nGenerating comparison report...")
    all_results = generate_comparison_report(ablation_results, dann_results, data_dir, fold_numbers)
    
    # Create final comparison figure
    print("\nCreating comparison figures...")
    create_final_comparison_figure(all_results, data_dir, fold_numbers)
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {os.path.join(data_dir, 'ablation_results.mat')}")
    print(f"  - {os.path.join(data_dir, 'ablation_report.txt')}")
    print(f"  - {os.path.join(data_dir, 'ablation_comparison.png')}")
    print(f"  - {os.path.join(data_dir, 'ablation_final_comparison.png')}")

if __name__ == "__main__":
    main()

