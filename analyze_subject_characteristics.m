function analyze_subject_characteristics()
% analyze_subject_characteristics() - Analyze per-subject characteristics 
% to understand fold performance variation.
%
% This script computes per-subject metrics that may explain model 
% generalization differences across held-out subjects.
%
% Outputs:
%   - diagnostics/subject_analysis_report.txt
%   - diagnostics/subject_analysis.mat
%   - Various visualization plots in diagnostics/analysis/
%
% Usage:
%   >> analyze_subject_characteristics()

%% Setup
fprintf('=== Subject Characteristics Analysis ===\n');

% Create output directories
if ~exist('diagnostics/analysis', 'dir')
    mkdir('diagnostics/analysis');
end

% Initialize results structure
results = struct();

%% Load Python Results First (to get actual accuracies)
fprintf('\n1. Loading Python training results...\n');
data_dir = 'python_data_SADT+SEED-VIG';  % Updated path for combined dataset

results_file = fullfile(data_dir, 'final_results.mat');
if ~exist(results_file, 'file')
    error('Results file not found: %s\nRun step4_domain_adversarial_training.py first.', results_file);
end

% Load results (Python savemat uses v5 format by default)
try
    loaded = load(results_file);
    
    fold_accuracies_raw = loaded.fold_accuracies;
    fold_numbers = loaded.fold_numbers;
    mean_accuracy = loaded.mean_accuracy;
    std_accuracy = loaded.std_accuracy;
    
    % Convert to column vector and percentages
    fold_accuracies = fold_accuracies_raw(:) * 100;
    fold_numbers = fold_numbers(:);
    num_folds = length(fold_numbers);
    
    fprintf('  Loaded results for %d folds\n', num_folds);
    fprintf('  Mean accuracy: %.2f%% ± %.2f%%\n', mean_accuracy * 100, std_accuracy * 100);
catch ME
    error('Error loading results: %s', ME.message);
end

%% Load Metadata
fprintf('\n2. Loading metadata...\n');
metadata_file = fullfile(data_dir, 'metadata.mat');
try
    % Try h5read first (MATLAB v7.3 format)
    num_subjects = double(h5read(metadata_file, '/num_subjects'));
    fprintf('  Total subjects: %d\n', num_subjects);
catch
    % Fall back to regular load
    try
        meta = load(metadata_file);
        num_subjects = double(meta.num_subjects);
        fprintf('  Total subjects: %d\n', num_subjects);
    catch ME
        warning('Could not load metadata: %s', ME.message);
        num_subjects = num_folds;  % Assume one fold per subject
    end
end

results.num_subjects = num_subjects;
results.num_folds = num_folds;

%% Load Per-Fold Data
fprintf('\n3. Loading fold data from Python exports...\n');

subject_data = struct();
fold_to_subject = containers.Map('KeyType', 'double', 'ValueType', 'double');

for i = 1:num_folds
    fold = fold_numbers(i);
    fold_file = fullfile(data_dir, sprintf('fold_%d_data.mat', fold));
    
    if ~exist(fold_file, 'file')
        warning('Fold %d data not found: %s', fold, fold_file);
        continue;
    end
    
    % Load using h5 format (Python exported MATLAB v7.3)
    try
        test_subject_nums = h5read(fold_file, '/test_subject_nums');
        YTest = h5read(fold_file, '/YTest_numeric');
        
        test_subject = mode(test_subject_nums);
        fold_to_subject(fold) = test_subject;
        
        % Class balance for this subject
        alert_count = sum(YTest == 1);
        drowsy_count = sum(YTest == 2);
        total_samples = length(YTest);
        
        % Find accuracy for this fold
        fold_idx = find(fold_numbers == fold);
        if ~isempty(fold_idx)
            acc = fold_accuracies(fold_idx);
        else
            acc = NaN;
        end
        
        subject_data(test_subject).subject_id = test_subject;
        subject_data(test_subject).fold = fold;
        subject_data(test_subject).total_samples = total_samples;
        subject_data(test_subject).alert_samples = alert_count;
        subject_data(test_subject).drowsy_samples = drowsy_count;
        subject_data(test_subject).class_balance_ratio = drowsy_count / max(alert_count, 1);
        subject_data(test_subject).minority_class_pct = 100 * min(alert_count, drowsy_count) / total_samples;
        subject_data(test_subject).test_accuracy = acc;
        
        fprintf('  Subject %2d (Fold %2d): %d samples (Alert: %d, Drowsy: %d) -> %.1f%% acc\n', ...
            test_subject, fold, total_samples, alert_count, drowsy_count, acc);
    catch ME
        warning('Error loading fold %d: %s', fold, ME.message);
    end
end

results.class_balance = subject_data;
results.fold_to_subject = fold_to_subject;

%% Compute Spectral Features Per Subject
fprintf('\n4. Computing spectral features per subject...\n');

% Define frequency bands
bands = struct();
bands.delta = [0.5 4];
bands.theta = [4 8];
bands.alpha = [8 13];
bands.beta = [13 30];

spectral_features = struct();

for i = 1:num_folds
    fold = fold_numbers(i);
    
    if ~isKey(fold_to_subject, fold)
        continue;
    end
    subj = fold_to_subject(fold);
    
    fold_file = fullfile(data_dir, sprintf('fold_%d_data.mat', fold));
    
    try
        % Load test data (spectral features)
        XTest = h5read(fold_file, '/XTest');  % Shape: [freq x channels x time x samples]
        YTest = h5read(fold_file, '/YTest_numeric');
        
        % XTest is [128 freq bins x 7 channels x 1 time x N samples]
        % Compute average power per frequency band per class
        
        % Assuming 128 frequency bins from 0 to 64 Hz (Nyquist for 128 Hz sampling)
        freq_bins = linspace(0, 64, 128);
        
        for band_name = fieldnames(bands)'
            band = bands.(band_name{1});
            band_indices = find(freq_bins >= band(1) & freq_bins < band(2));
            
            if isempty(band_indices)
                continue;
            end
            
            % Average power in this band across all channels and samples
            band_power = squeeze(mean(XTest(band_indices, :, :, :), [1, 2, 3]));  % [N samples]
            
            % Split by class
            alert_power = band_power(YTest == 1);
            drowsy_power = band_power(YTest == 2);
            
            if ~isempty(alert_power) && ~isempty(drowsy_power)
                spectral_features(subj).(band_name{1}).mean_alert = mean(alert_power);
                spectral_features(subj).(band_name{1}).mean_drowsy = mean(drowsy_power);
                spectral_features(subj).(band_name{1}).std_alert = std(alert_power);
                spectral_features(subj).(band_name{1}).std_drowsy = std(drowsy_power);
                
                pooled_std = sqrt((std(alert_power)^2 + std(drowsy_power)^2) / 2);
                if pooled_std > 0
                    spectral_features(subj).(band_name{1}).effect_size = ...
                        (mean(drowsy_power) - mean(alert_power)) / pooled_std;
                else
                    spectral_features(subj).(band_name{1}).effect_size = 0;
                end
            end
        end
        
        fprintf('  Subject %2d: Spectral features computed\n', subj);
    catch ME
        warning('Error computing spectral features for subject %d: %s', subj, ME.message);
    end
end

results.spectral_features = spectral_features;

%% Compute Signal Quality Metrics
fprintf('\n5. Computing signal quality metrics...\n');

quality_metrics = struct();

for i = 1:num_folds
    fold = fold_numbers(i);
    
    if ~isKey(fold_to_subject, fold)
        continue;
    end
    subj = fold_to_subject(fold);
    
    fold_file = fullfile(data_dir, sprintf('fold_%d_data.mat', fold));
    
    try
        XTest = h5read(fold_file, '/XTest');  % [freq x channels x time x samples]
        
        % Compute signal variability metrics
        all_power = squeeze(mean(XTest, [1, 2, 3]));  % Average power per sample
        
        quality_metrics(subj).mean_power = mean(all_power);
        quality_metrics(subj).std_power = std(all_power);
        quality_metrics(subj).cv_power = std(all_power) / max(mean(all_power), eps);  % Coefficient of variation
        quality_metrics(subj).snr_estimate = mean(all_power) / max(std(all_power), eps);
        
        % Per-channel variability
        channel_power = squeeze(mean(XTest, [1, 3, 4]));  % [7 channels]
        quality_metrics(subj).channel_variability = std(channel_power) / max(mean(channel_power), eps);
        
        fprintf('  Subject %2d: SNR=%.2f, CV=%.3f, Chan_Var=%.3f\n', ...
            subj, quality_metrics(subj).snr_estimate, quality_metrics(subj).cv_power, ...
            quality_metrics(subj).channel_variability);
    catch ME
        warning('Error computing quality metrics for subject %d: %s', subj, ME.message);
    end
end

results.quality_metrics = quality_metrics;
results.fold_accuracies = fold_accuracies;
results.fold_numbers = fold_numbers;

%% Statistical Analysis
fprintf('\n6. Performing statistical analysis...\n');

% Categorize folds by performance
high_performers = fold_numbers(fold_accuracies >= 95);
mid_performers = fold_numbers(fold_accuracies >= 70 & fold_accuracies < 95);
low_performers = fold_numbers(fold_accuracies < 70);

fprintf('  High performers (≥95%%): %d folds\n', length(high_performers));
fprintf('  Mid performers (70-95%%): %d folds\n', length(mid_performers));
fprintf('  Low performers (<70%%): %d folds\n', length(low_performers));

results.performance_groups = struct();
results.performance_groups.high_folds = high_performers;
results.performance_groups.mid_folds = mid_performers;
results.performance_groups.low_folds = low_performers;

%% Generate Visualizations
fprintf('\n7. Generating visualizations...\n');

generate_visualizations(subject_data, quality_metrics, spectral_features, ...
    fold_accuracies, fold_numbers, fold_to_subject);

%% Generate Report
fprintf('\n8. Generating text report...\n');

generate_text_report(results, subject_data, quality_metrics, spectral_features, ...
    fold_accuracies, fold_numbers, fold_to_subject);

%% Save Results
fprintf('\n9. Saving results...\n');
save('diagnostics/subject_analysis.mat', 'results', '-v7.3');
fprintf('Results saved to: diagnostics/subject_analysis.mat\n');

fprintf('\n=== Analysis Complete ===\n');
fprintf('Report saved to: diagnostics/subject_analysis_report.txt\n');
fprintf('Figures saved to: diagnostics/analysis/\n');

end

%% Helper Functions

function generate_visualizations(subject_data, quality_metrics, spectral_features, ...
    fold_accuracies, fold_numbers, fold_to_subject)
    
    % Extract subject IDs and metrics
    subjects = [];
    class_ratios = [];
    minority_pcts = [];
    snr_vals = [];
    cv_vals = [];
    accuracies = [];
    
    for i = 1:length(fold_numbers)
        fold = fold_numbers(i);
        if ~isKey(fold_to_subject, fold)
            continue;
        end
        subj = fold_to_subject(fold);
        
        if length(subject_data) >= subj && isfield(subject_data(subj), 'subject_id') && ~isempty(subject_data(subj).subject_id)
            subjects(end+1) = subj;
            class_ratios(end+1) = subject_data(subj).class_balance_ratio;
            minority_pcts(end+1) = subject_data(subj).minority_class_pct;
            accuracies(end+1) = subject_data(subj).test_accuracy;
            
            if length(quality_metrics) >= subj && isfield(quality_metrics(subj), 'snr_estimate')
                snr_vals(end+1) = quality_metrics(subj).snr_estimate;
                cv_vals(end+1) = quality_metrics(subj).cv_power;
            else
                snr_vals(end+1) = NaN;
                cv_vals(end+1) = NaN;
            end
        end
    end
    
    if isempty(subjects)
        warning('No subject data available for visualization');
        return;
    end
    
    % Figure 1: Class Balance vs Accuracy
    figure('Position', [100, 100, 1200, 400], 'Visible', 'off');
    
    subplot(1, 3, 1);
    scatter(class_ratios, accuracies, 100, 'filled');
    xlabel('Drowsy/Alert Ratio');
    ylabel('Test Accuracy (%)');
    title('Class Balance vs Accuracy');
    grid on;
    
    subplot(1, 3, 2);
    scatter(minority_pcts, accuracies, 100, 'filled');
    xlabel('Minority Class (%)');
    ylabel('Test Accuracy (%)');
    title('Class Imbalance vs Accuracy');
    grid on;
    
    subplot(1, 3, 3);
    bar(1:length(subjects), accuracies);
    xlabel('Subject Index');
    ylabel('Test Accuracy (%)');
    title('Per-Subject Performance');
    ylim([0 105]);
    grid on;
    
    saveas(gcf, 'diagnostics/analysis/class_balance_analysis.png');
    close(gcf);
    
    % Figure 2: Signal Quality vs Accuracy
    figure('Position', [100, 100, 1200, 400], 'Visible', 'off');
    
    subplot(1, 3, 1);
    valid_snr = ~isnan(snr_vals);
    if any(valid_snr)
        scatter(snr_vals(valid_snr), accuracies(valid_snr), 100, 'filled');
        xlabel('SNR Estimate');
        ylabel('Test Accuracy (%)');
        title('Signal Quality vs Accuracy');
        grid on;
    end
    
    subplot(1, 3, 2);
    valid_cv = ~isnan(cv_vals);
    if any(valid_cv)
        scatter(cv_vals(valid_cv), accuracies(valid_cv), 100, 'filled');
        xlabel('Coefficient of Variation');
        ylabel('Test Accuracy (%)');
        title('Signal Variability vs Accuracy');
        grid on;
    end
    
    subplot(1, 3, 3);
    % Categorize subjects by color
    colors = zeros(length(subjects), 3);
    for i = 1:length(subjects)
        if accuracies(i) >= 95
            colors(i, :) = [0 0.8 0];  % Green
        elseif accuracies(i) >= 70
            colors(i, :) = [1 0.6 0];  % Orange
        else
            colors(i, :) = [0.8 0 0];  % Red
        end
    end
    scatter(1:length(subjects), accuracies, 100, colors, 'filled');
    xlabel('Subject Index');
    ylabel('Test Accuracy (%)');
    title('Performance Categories');
    ylim([0 105]);
    grid on;
    
    saveas(gcf, 'diagnostics/analysis/signal_quality_analysis.png');
    close(gcf);
    
    % Figure 3: Spectral Features
    if ~isempty(fieldnames(spectral_features))
        figure('Position', [100, 100, 1200, 800], 'Visible', 'off');
        
        bands = {'theta', 'alpha', 'beta'};
        for b = 1:length(bands)
            band = bands{b};
            
            effect_sizes = [];
            accs = [];
            
            for i = 1:length(fold_numbers)
                fold = fold_numbers(i);
                if ~isKey(fold_to_subject, fold)
                    continue;
                end
                subj = fold_to_subject(fold);
                
                if length(spectral_features) >= subj && isfield(spectral_features(subj), band) && ...
                   isfield(spectral_features(subj).(band), 'effect_size')
                    
                    effect_sizes(end+1) = spectral_features(subj).(band).effect_size;
                    accs(end+1) = subject_data(subj).test_accuracy;
                end
            end
            
            if ~isempty(effect_sizes)
                subplot(2, 2, b);
                scatter(effect_sizes, accs, 100, 'filled');
                xlabel(sprintf('%s Band Effect Size (Cohen''s d)', band));
                ylabel('Test Accuracy (%)');
                title(sprintf('%s Band: Alert vs Drowsy', band));
                grid on;
                
                % Add correlation
                if length(effect_sizes) > 2
                    [r, p] = corr(effect_sizes', accs');
                    text(min(effect_sizes), max(accs), sprintf('r=%.2f, p=%.3f', r, p), ...
                        'VerticalAlignment', 'top');
                end
            end
        end
        
        saveas(gcf, 'diagnostics/analysis/spectral_features_analysis.png');
        close(gcf);
    end
    
    % Figure 4: Effect Size vs Accuracy (single alpha band)
    figure('Position', [100, 100, 800, 600], 'Visible', 'off');
    
    all_effect_sizes = [];
    all_accs = [];
    
    for i = 1:length(fold_numbers)
        fold = fold_numbers(i);
        if ~isKey(fold_to_subject, fold)
            continue;
        end
        subj = fold_to_subject(fold);
        
        if length(spectral_features) >= subj && isfield(spectral_features(subj), 'alpha') && ...
           isfield(spectral_features(subj).alpha, 'effect_size')
            all_effect_sizes(end+1) = abs(spectral_features(subj).alpha.effect_size);
            all_accs(end+1) = subject_data(subj).test_accuracy;
        end
    end
    
    if ~isempty(all_effect_sizes)
        scatter(all_effect_sizes, all_accs, 100, 'filled');
        xlabel('Alpha Band |Effect Size| (Cohen''s d)');
        ylabel('Test Accuracy (%)');
        title('Spectral Separability vs Model Performance');
        grid on;
        
        if length(all_effect_sizes) > 2
            [r, p] = corr(all_effect_sizes', all_accs');
            text(min(all_effect_sizes), max(all_accs), sprintf('r=%.2f, p=%.3f', r, p), ...
                'VerticalAlignment', 'top', 'FontSize', 12);
        end
    end
    
    saveas(gcf, 'diagnostics/analysis/effect_size_vs_accuracy.png');
    print(gcf, 'diagnostics/analysis/effect_size_vs_accuracy.pdf', '-dpdf', '-bestfit');
    close(gcf);
    
    % Figure 5: Combined Band Analysis (publication-quality two-panel figure)
    generate_combined_effect_size_figure(subject_data, spectral_features, fold_numbers, fold_to_subject);
    
    % Generate spectral analysis text report
    generate_spectral_analysis_report(subject_data, spectral_features, fold_numbers, fold_to_subject);
    
    fprintf('  Visualizations saved to diagnostics/analysis/\n');
end

function generate_combined_effect_size_figure(subject_data, spectral_features, fold_numbers, fold_to_subject)
    % Create publication-quality two-panel figure (stacked for IEEE double column)
    
    % Collect data for all subjects
    subjects_list = [];
    accuracies = [];
    theta_es = [];
    alpha_es = [];
    beta_es = [];
    
    for i = 1:length(fold_numbers)
        fold = fold_numbers(i);
        if ~isKey(fold_to_subject, fold)
            continue;
        end
        subj = fold_to_subject(fold);
        
        if length(spectral_features) >= subj && isfield(spectral_features(subj), 'theta') && ...
           isfield(spectral_features(subj), 'alpha') && isfield(spectral_features(subj), 'beta')
            
            subjects_list(end+1) = subj;
            accuracies(end+1) = subject_data(subj).test_accuracy;
            theta_es(end+1) = spectral_features(subj).theta.effect_size;
            alpha_es(end+1) = spectral_features(subj).alpha.effect_size;
            beta_es(end+1) = spectral_features(subj).beta.effect_size;
        end
    end
    
    if isempty(subjects_list)
        warning('No spectral data available for combined figure');
        return;
    end
    
    % Compute mean absolute effect size across bands
    mean_abs_es = (abs(theta_es) + abs(alpha_es) + abs(beta_es)) / 3;
    
    % Categorize by performance (using 90% threshold to match example)
    high_mask = accuracies >= 90;
    mid_mask = accuracies >= 70 & accuracies < 90;
    low_mask = accuracies < 70;
    
    % Create figure - stacked layout for IEEE double column
    fig = figure('Position', [100, 100, 450, 800], 'Visible', 'off');
    
    % Panel A: Combined Band Analysis scatter plot
    ax1 = subplot(2, 1, 1);
    hold on;
    
    % Plot by category with different colors (no subject labels)
    if any(low_mask)
        scatter(mean_abs_es(low_mask), accuracies(low_mask), 60, [0.8 0.2 0.2], 'filled', 'DisplayName', 'Low (<70%)');
    end
    if any(mid_mask)
        scatter(mean_abs_es(mid_mask), accuracies(mid_mask), 60, [1 0.6 0], 'filled', 'DisplayName', 'Mid (70-90%)');
    end
    if any(high_mask)
        scatter(mean_abs_es(high_mask), accuracies(high_mask), 60, [0.2 0.7 0.2], 'filled', 'DisplayName', 'High (>=90%)');
    end
    
    % Add trend line
    if length(mean_abs_es) > 2
        [r, p] = corr(mean_abs_es', accuracies');
        coeffs = polyfit(mean_abs_es, accuracies, 1);
        x_fit = linspace(min(mean_abs_es), max(mean_abs_es), 100);
        y_fit = polyval(coeffs, x_fit);
        plot(x_fit, y_fit, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
    
    xlabel('Mean Absolute Effect Size |d|', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Test Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title('(A) Combined Band Analysis', 'FontSize', 14, 'FontWeight', 'bold');
    leg1 = legend('Location', 'southeast', 'FontSize', 10);
    set(leg1, 'Box', 'on');
    set(ax1, 'FontSize', 11);
    grid on;
    set(ax1, 'GridLineStyle', '-', 'GridAlpha', 0.3);
    xlim([0 max(mean_abs_es)*1.15]);
    ylim([55 100]);
    hold off;
    
    % Panel B: Group Comparison bar chart
    ax2 = subplot(2, 1, 2);
    
    % Compute group statistics
    bands = {'Theta', 'Alpha', 'Beta'};
    low_means = [mean(abs(theta_es(low_mask))), mean(abs(alpha_es(low_mask))), mean(abs(beta_es(low_mask)))];
    low_stds = [std(abs(theta_es(low_mask))), std(abs(alpha_es(low_mask))), std(abs(beta_es(low_mask)))];
    high_means = [mean(abs(theta_es(high_mask))), mean(abs(alpha_es(high_mask))), mean(abs(beta_es(high_mask)))];
    high_stds = [std(abs(theta_es(high_mask))), std(abs(alpha_es(high_mask))), std(abs(beta_es(high_mask)))];
    
    % Handle NaN for groups with no subjects
    low_means(isnan(low_means)) = 0;
    low_stds(isnan(low_stds)) = 0;
    high_means(isnan(high_means)) = 0;
    high_stds(isnan(high_stds)) = 0;
    
    x = 1:3;
    width = 0.35;
    
    hold on;
    b1 = bar(x - width/2, low_means, width, 'FaceColor', [0.8 0.3 0.3], 'DisplayName', 'Low Performers (<70%)');
    b2 = bar(x + width/2, high_means, width, 'FaceColor', [0.3 0.7 0.3], 'DisplayName', 'High Performers (>=90%)');
    
    % Add error bars
    errorbar(x - width/2, low_means, low_stds, 'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    errorbar(x + width/2, high_means, high_stds, 'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    
    set(gca, 'XTick', x, 'XTickLabel', bands);
    xlabel('Frequency Band', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Mean Absolute Effect Size |d|', 'FontSize', 12, 'FontWeight', 'bold');
    title('(B) Group Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    leg2 = legend('Location', 'northwest', 'FontSize', 10);
    set(leg2, 'Box', 'on');
    set(ax2, 'FontSize', 11);
    grid on;
    set(ax2, 'GridLineStyle', '-', 'GridAlpha', 0.3);
    hold off;
    
    % Adjust subplot spacing (with padding for titles)
    set(ax1, 'Position', [0.15 0.56 0.78 0.36]);
    set(ax2, 'Position', [0.15 0.10 0.78 0.36]);
    
    % Add extra padding between title and plot
    set(ax1.Title, 'Units', 'normalized', 'Position', [0.5, 1.08, 0]);
    set(ax2.Title, 'Units', 'normalized', 'Position', [0.5, 1.08, 0]);
    
    % Save figure
    saveas(fig, 'diagnostics/analysis/effect_size_vs_accuracy_combined.png');
    print(fig, 'diagnostics/analysis/effect_size_vs_accuracy_combined.pdf', '-dpdf', '-bestfit');
    close(fig);
    
    fprintf('  Combined effect size figure saved (PNG + PDF)\n');
end

function generate_spectral_analysis_report(subject_data, spectral_features, fold_numbers, fold_to_subject)
    % Generate detailed spectral analysis text report
    
    fid = fopen('diagnostics/analysis/spectral_analysis.txt', 'w');
    
    % Collect data
    subjects_list = [];
    accuracies = [];
    theta_es = [];
    alpha_es = [];
    beta_es = [];
    
    for i = 1:length(fold_numbers)
        fold = fold_numbers(i);
        if ~isKey(fold_to_subject, fold)
            continue;
        end
        subj = fold_to_subject(fold);
        
        if length(spectral_features) >= subj && isfield(spectral_features(subj), 'theta') && ...
           isfield(spectral_features(subj), 'alpha') && isfield(spectral_features(subj), 'beta')
            
            subjects_list(end+1) = subj;
            accuracies(end+1) = subject_data(subj).test_accuracy;
            theta_es(end+1) = spectral_features(subj).theta.effect_size;
            alpha_es(end+1) = spectral_features(subj).alpha.effect_size;
            beta_es(end+1) = spectral_features(subj).beta.effect_size;
        end
    end
    
    % Categorize
    low_mask = accuracies < 70;
    high_mask = accuracies >= 95;
    
    % Low performers section
    fprintf(fid, '=== LOW PERFORMERS (<70%%) ===\n\n');
    low_indices = find(low_mask);
    for idx = low_indices
        fprintf(fid, 'Subject %d:\n', subjects_list(idx));
        fprintf(fid, '  Test Accuracy: %.1f%%\n', accuracies(idx));
        fprintf(fid, '  Theta effect size: %.3f\n', theta_es(idx));
        fprintf(fid, '  Alpha effect size: %.3f\n', alpha_es(idx));
        fprintf(fid, '  Beta effect size: %.3f\n\n', beta_es(idx));
    end
    
    % High performers section
    fprintf(fid, '=== HIGH PERFORMERS (≥95%%) ===\n\n');
    high_indices = find(high_mask);
    for idx = high_indices
        fprintf(fid, 'Subject %d:\n', subjects_list(idx));
        fprintf(fid, '  Test Accuracy: %.1f%%\n', accuracies(idx));
        fprintf(fid, '  Theta effect size: %.3f\n', theta_es(idx));
        fprintf(fid, '  Alpha effect size: %.3f\n', alpha_es(idx));
        fprintf(fid, '  Beta effect size: %.3f\n\n', beta_es(idx));
    end
    
    % Statistical comparison
    fprintf(fid, '=== STATISTICAL COMPARISON ===\n');
    
    if any(low_mask) && any(high_mask)
        fprintf(fid, 'Theta effect size:\n');
        fprintf(fid, '  Low performers: %.3f ± %.3f\n', mean(theta_es(low_mask)), std(theta_es(low_mask)));
        fprintf(fid, '  High performers: %.3f ± %.3f\n', mean(theta_es(high_mask)), std(theta_es(high_mask)));
        
        fprintf(fid, 'Alpha effect size:\n');
        fprintf(fid, '  Low performers: %.3f ± %.3f\n', mean(alpha_es(low_mask)), std(alpha_es(low_mask)));
        fprintf(fid, '  High performers: %.3f ± %.3f\n', mean(alpha_es(high_mask)), std(alpha_es(high_mask)));
        
        fprintf(fid, 'Beta effect size:\n');
        fprintf(fid, '  Low performers: %.3f ± %.3f\n', mean(beta_es(low_mask)), std(beta_es(low_mask)));
        fprintf(fid, '  High performers: %.3f ± %.3f\n', mean(beta_es(high_mask)), std(beta_es(high_mask)));
    else
        fprintf(fid, 'Insufficient data for group comparison.\n');
    end
    
    % Correlation analysis
    fprintf(fid, '\n=== CORRELATION ANALYSIS ===\n');
    mean_abs_es = (abs(theta_es) + abs(alpha_es) + abs(beta_es)) / 3;
    
    if length(accuracies) > 2
        [r_combined, p_combined] = corr(mean_abs_es', accuracies');
        [r_theta, p_theta] = corr(abs(theta_es)', accuracies');
        [r_alpha, p_alpha] = corr(abs(alpha_es)', accuracies');
        [r_beta, p_beta] = corr(abs(beta_es)', accuracies');
        
        fprintf(fid, 'Combined bands vs Accuracy: r=%.3f, p=%.4f\n', r_combined, p_combined);
        fprintf(fid, 'Theta |d| vs Accuracy: r=%.3f, p=%.4f\n', r_theta, p_theta);
        fprintf(fid, 'Alpha |d| vs Accuracy: r=%.3f, p=%.4f\n', r_alpha, p_alpha);
        fprintf(fid, 'Beta |d| vs Accuracy: r=%.3f, p=%.4f\n', r_beta, p_beta);
    end
    
    fclose(fid);
    fprintf('  Spectral analysis report saved to diagnostics/analysis/spectral_analysis.txt\n');
    
    % === Generate TABLE III format file ===
    fid_table = fopen('diagnostics/analysis/table_iii_effect_sizes.txt', 'w');
    
    fprintf(fid_table, 'TABLE III: SPECTRAL EFFECT SIZES BY PERFORMANCE GROUP\n');
    fprintf(fid_table, '=========================================================\n\n');
    fprintf(fid_table, 'Band\t\tHigh Performers\t\tLow Performers\t\tp-value\n');
    fprintf(fid_table, '---------------------------------------------------------\n');
    
    % Use 90% threshold for high performers to match figure
    high_mask_90 = accuracies >= 90;
    n_high = sum(high_mask_90);
    n_low = sum(low_mask);
    
    if any(low_mask) && any(high_mask_90)
        % Compute t-test p-values (two-sample, unequal variance - Welch's t-test)
        [~, p_theta_ttest] = ttest2(abs(theta_es(high_mask_90)), abs(theta_es(low_mask)), 'Vartype', 'unequal');
        [~, p_alpha_ttest] = ttest2(abs(alpha_es(high_mask_90)), abs(alpha_es(low_mask)), 'Vartype', 'unequal');
        [~, p_beta_ttest] = ttest2(abs(beta_es(high_mask_90)), abs(beta_es(low_mask)), 'Vartype', 'unequal');
        
        fprintf(fid_table, 'Theta (θ)\t%.2f ± %.2f\t\t\t%.2f ± %.2f\t\t\tp=%.3f\n', ...
            mean(abs(theta_es(high_mask_90))), std(abs(theta_es(high_mask_90))), ...
            mean(abs(theta_es(low_mask))), std(abs(theta_es(low_mask))), p_theta_ttest);
        fprintf(fid_table, 'Alpha (α)\t%.2f ± %.2f\t\t\t%.2f ± %.2f\t\t\tp=%.3f\n', ...
            mean(abs(alpha_es(high_mask_90))), std(abs(alpha_es(high_mask_90))), ...
            mean(abs(alpha_es(low_mask))), std(abs(alpha_es(low_mask))), p_alpha_ttest);
        fprintf(fid_table, 'Beta (β)\t%.2f ± %.2f\t\t\t%.2f ± %.2f\t\t\tp=%.3f\n', ...
            mean(abs(beta_es(high_mask_90))), std(abs(beta_es(high_mask_90))), ...
            mean(abs(beta_es(low_mask))), std(abs(beta_es(low_mask))), p_beta_ttest);
    else
        fprintf(fid_table, 'Insufficient data for group comparison.\n');
    end
    
    fprintf(fid_table, '---------------------------------------------------------\n\n');
    fprintf(fid_table, 'High Performers: ≥90%% accuracy (n=%d)\n', n_high);
    fprintf(fid_table, 'Low Performers:  <70%% accuracy (n=%d)\n', n_low);
    fprintf(fid_table, 'Statistical test: Two-sample t-test (unequal variance assumed)\n\n');
    
    % Add correlation analysis
    fprintf(fid_table, '=========================================================\n');
    fprintf(fid_table, 'CORRELATION ANALYSIS: Effect Size vs Accuracy\n');
    fprintf(fid_table, '=========================================================\n\n');
    
    if length(accuracies) > 2
        fprintf(fid_table, 'Band\t\t\tCorrelation (r)\t\tp-value\n');
        fprintf(fid_table, '---------------------------------------------------------\n');
        fprintf(fid_table, 'Theta |d|\t\t%.3f\t\t\t\tp=%.4f\n', r_theta, p_theta);
        fprintf(fid_table, 'Alpha |d|\t\t%.3f\t\t\t\tp=%.4f\n', r_alpha, p_alpha);
        fprintf(fid_table, 'Beta |d|\t\t%.3f\t\t\t\tp=%.4f\n', r_beta, p_beta);
        fprintf(fid_table, 'Combined |d|\t%.3f\t\t\t\tp=%.4f\n', r_combined, p_combined);
        fprintf(fid_table, '---------------------------------------------------------\n\n');
        fprintf(fid_table, 'Note: Correlation computed using Pearson''s r between\n');
        fprintf(fid_table, '      absolute effect sizes and test accuracy.\n');
    end
    
    fclose(fid_table);
    fprintf('  Table III effect sizes saved to diagnostics/analysis/table_iii_effect_sizes.txt\n');
end

function generate_text_report(results, subject_data, quality_metrics, spectral_features, ...
    fold_accuracies, fold_numbers, fold_to_subject)
    
    fid = fopen('diagnostics/subject_analysis_report.txt', 'w');
    
    fprintf(fid, '================================================================================\n');
    fprintf(fid, '           SUBJECT CHARACTERISTICS ANALYSIS REPORT\n');
    fprintf(fid, '           EEG Drowsiness Detection - Cross-Subject Validation\n');
    fprintf(fid, '           Combined SADT + SEED-VIG Dataset\n');
    fprintf(fid, '================================================================================\n\n');
    
    fprintf(fid, 'OVERALL PERFORMANCE SUMMARY\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, 'Number of subjects: %d\n', results.num_subjects);
    fprintf(fid, 'Number of folds processed: %d\n', length(fold_numbers));
    fprintf(fid, 'Mean Accuracy: %.2f%% ± %.2f%%\n', mean(fold_accuracies), std(fold_accuracies));
    fprintf(fid, 'Median Accuracy: %.2f%%\n', median(fold_accuracies));
    fprintf(fid, 'Range: %.2f%% - %.2f%%\n', min(fold_accuracies), max(fold_accuracies));
    fprintf(fid, 'Coefficient of Variation: %.2f%%\n\n', 100 * std(fold_accuracies) / mean(fold_accuracies));
    
    fprintf(fid, 'PERFORMANCE DISTRIBUTION\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    high_count = sum(fold_accuracies >= 95);
    mid_count = sum(fold_accuracies >= 70 & fold_accuracies < 95);
    low_count = sum(fold_accuracies < 70);
    
    fprintf(fid, 'High performers (≥95%%): %d subjects (%.1f%%)\n', high_count, 100*high_count/length(fold_numbers));
    fprintf(fid, 'Mid performers (70-95%%): %d subjects (%.1f%%)\n', mid_count, 100*mid_count/length(fold_numbers));
    fprintf(fid, 'Low performers (<70%%): %d subjects (%.1f%%)\n\n', low_count, 100*low_count/length(fold_numbers));
    
    fprintf(fid, 'PER-FOLD RESULTS\n');
    fprintf(fid, '================================================================================\n\n');
    
    % Sort by accuracy for better readability
    [sorted_acc, sort_idx] = sort(fold_accuracies, 'descend');
    sorted_folds = fold_numbers(sort_idx);
    
    for i = 1:length(sorted_folds)
        fold = sorted_folds(i);
        acc = sorted_acc(i);
        
        if ~isKey(fold_to_subject, fold)
            continue;
        end
        subj = fold_to_subject(fold);
        
        % Performance category
        if acc >= 95
            category = 'HIGH';
        elseif acc >= 70
            category = 'MID';
        else
            category = 'LOW';
        end
        
        fprintf(fid, 'Fold %2d (Subject %2d) - %s: %.2f%%\n', fold, subj, category, acc);
        
        if length(subject_data) >= subj && isfield(subject_data(subj), 'total_samples')
            fprintf(fid, '  Samples: %d (Alert: %d, Drowsy: %d, Ratio: %.2f)\n', ...
                subject_data(subj).total_samples, ...
                subject_data(subj).alert_samples, ...
                subject_data(subj).drowsy_samples, ...
                subject_data(subj).class_balance_ratio);
        end
        
        if length(quality_metrics) >= subj && isfield(quality_metrics(subj), 'snr_estimate')
            fprintf(fid, '  Signal: SNR=%.2f, CV=%.3f\n', ...
                quality_metrics(subj).snr_estimate, ...
                quality_metrics(subj).cv_power);
        end
        
        if length(spectral_features) >= subj && isfield(spectral_features(subj), 'alpha')
            fprintf(fid, '  Effect sizes: theta=%.2f, alpha=%.2f, beta=%.2f\n', ...
                spectral_features(subj).theta.effect_size, ...
                spectral_features(subj).alpha.effect_size, ...
                spectral_features(subj).beta.effect_size);
        end
        
        fprintf(fid, '\n');
    end
    
    fprintf(fid, 'KEY FINDINGS\n');
    fprintf(fid, '================================================================================\n\n');
    
    fprintf(fid, '1. INTER-SUBJECT VARIABILITY:\n');
    fprintf(fid, '   Performance varied substantially across held-out subjects, ranging from\n');
    fprintf(fid, '   %.1f%% to %.1f%%. This %.1f-percentage-point range indicates significant\n', ...
        min(fold_accuracies), max(fold_accuracies), max(fold_accuracies) - min(fold_accuracies));
    fprintf(fid, '   inter-subject variance, consistent with published cross-subject EEG studies.\n\n');
    
    fprintf(fid, '2. PERFORMANCE CATEGORIES:\n');
    fprintf(fid, '   The model generalized excellently to %d subjects (≥95%% accuracy),\n', high_count);
    fprintf(fid, '   moderately to %d subjects (70-95%%), and poorly to %d subjects (<70%%).\n\n', ...
        mid_count, low_count);
    
    fprintf(fid, '3. VARIANCE ANALYSIS:\n');
    if std(fold_accuracies) > 15
        fprintf(fid, '   High variance (SD=%.1f%%) suggests substantial individual differences\n', std(fold_accuracies));
        fprintf(fid, '   in drowsiness manifestation or signal characteristics.\n\n');
    else
        fprintf(fid, '   Moderate variance (SD=%.1f%%) indicates reasonable consistency\n', std(fold_accuracies));
        fprintf(fid, '   across subjects with some individual differences.\n\n');
    end
    
    fprintf(fid, 'RECOMMENDED REPORTING LANGUAGE\n');
    fprintf(fid, '================================================================================\n\n');
    
    fprintf(fid, 'SAFE TO STATE (data-supported):\n');
    fprintf(fid, '- "Performance varied across held-out subjects (range: %.1f-%.1f%%)"\n', ...
        min(fold_accuracies), max(fold_accuracies));
    fprintf(fid, '- "The model generalized well to some subjects (≥95%%) but not all"\n');
    fprintf(fid, '- "Mean cross-validated accuracy was %.1f%% ± %.1f%%"\n', ...
        mean(fold_accuracies), std(fold_accuracies));
    fprintf(fid, '- "This pattern is consistent with other cross-subject EEG studies"\n\n');
    
    fprintf(fid, 'HYPOTHESIS-ONLY (requires additional analysis):\n');
    fprintf(fid, '- "Plausible explanations include individual differences in drowsiness\n');
    fprintf(fid, '   manifestation, signal quality, or class separability"\n');
    fprintf(fid, '- "Subjects with low separability in spectral features showed lower accuracy"\n\n');
    
    fprintf(fid, '================================================================================\n');
    fprintf(fid, 'End of Report\n');
    fprintf(fid, '================================================================================\n');
    
    fclose(fid);
    fprintf('  Report written to diagnostics/subject_analysis_report.txt\n');
end
