%% SEED-VLA/VRW DROWSINESS ANALYSIS PIPELINE
% This script preprocesses the SEED-VLA (lab) and SEED-VRW (real-world) 
% datasets for drowsiness detection using your existing preprocessing pipeline.
%
% Data Format:
%   - EEG: .edf files at 300 Hz, 24 channels (referenced to Pz)
%   - PERCLOS: .mat files with continuous vigilance labels (0-1)
%
% Preprocessing Pipeline:
%   1. Load EDF files and fix channel names (remove " - Pz" suffix)
%   2. Downsample to 250 Hz
%   3. Add channel locations
%   4. Bandpass filter (1-45 Hz)
%   5. vEOG regression (blink removal)
%   6. ICA + ICLabel artifact removal
%   7. Extract PERCLOS-labeled segments
%   8. Export for Python DANN training
%
% Labeling Options (configurable):
%   Option 1: Original SEED-VIG thresholds (Alert<0.35, Drowsy>0.70)
%   Option 2: Relaxed thresholds (Alert<0.40, Drowsy>0.60)
%   Option 3: Tertile-based (bottom 33% = Alert, top 33% = Drowsy)
%   Option 4: Continuous regression (use PERCLOS value directly)
%
% Usage:
%   1. Set labeling_option and other parameters below
%   2. Run this script in MATLAB with EEGLAB loaded
%
% Requires: EEGLAB, ICLabel plugin, biosig plugin (for EDF reading)

%% --- USER-DEFINED PARAMETERS ---

% Data paths
vla_eeg_path = '/Users/ziglicis/Desktop/Drowsiness/data/VLA_VRW/lab/EEG';
vla_perclos_path = '/Users/ziglicis/Desktop/Drowsiness/data/VLA_VRW/lab/perclos';
vrw_eeg_path = '/Users/ziglicis/Desktop/Drowsiness/data/VLA_VRW/real/EEG';
vrw_perclos_path = '/Users/ziglicis/Desktop/Drowsiness/data/VLA_VRW/real/perclos';

% Output directory
output_dir = 'diagnostics/python_data_vla_vrw';

% Preprocessing Parameters
low_cutoff_freq  = 1;     % High-pass filter cutoff (Hz)
high_cutoff_freq = 45;    % Low-pass filter cutoff (Hz)
downsample_rate  = 250;   % Target sampling rate (Hz)

% Segment Parameters
segment_duration_sec = 3;  % Duration of each segment (seconds)
skip_initial_sec = 24;     % Skip first N seconds (as per README)

% ========================================================================
% LABELING OPTIONS - Choose one:
% ========================================================================
% 1 = Original SEED-VIG thresholds (Alert < 0.35, Drowsy > 0.70)
%     - Strict separation, discards middle ~35% of data
%     - Best for clear-cut drowsiness detection
%     - Expected: ~8 valid subjects, ~1900 samples
%
% 2 = Relaxed thresholds (Alert < 0.40, Drowsy > 0.60)
%     - More inclusive, discards middle ~20% of data
%     - Good balance of data quantity and label quality
%     - Expected: ~15 valid subjects, ~3700 samples
%
% 3 = Median split (Alert < median, Drowsy > median)
%     - Uses each subject's own median PERCLOS as threshold
%     - Maximizes data usage, subject-normalized labels
%     - All subjects valid, no data discarded
%
% 4 = Tertile-based (bottom 33% = Alert, top 33% = Drowsy)
%     - Per-subject adaptive thresholds
%     - Discards middle 33%, balanced classes guaranteed
%     - All subjects with sufficient data valid
%
labeling_option = 2;  % <-- CHANGE THIS TO SELECT LABELING METHOD

% Minimum samples per class per subject (for options 1, 2, 4)
min_samples_per_class = 30;

% Visualization
enable_visualization = true;
viz_subject_limit = Inf;

% ========================================================================
% Channel Configuration
% ========================================================================
% Standard 10-20 channels available in VLA/VRW (18 channels)
% Note: Original names are like "EEG Fp1 - Pz", need to extract "Fp1"
standard_channels = {'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', ...
                     'C3', 'C4', 'Cz', 'T3', 'T4', 'T5', 'T6', ...
                     'P3', 'P4', 'O1', 'O2'};

% Reference channels (for linked mastoid reference if available)
% VLA/VRW has A1, A2 available
ref_channels = {'A1', 'A2'};

% Frontal channels for final output (matching your other pipelines)
frontal_channels = {'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'};

% --- END OF PARAMETERS ---

%% Initialize EEGLAB
if ~exist('eeglab', 'file')
    eeglab_path = fullfile(fileparts(pwd), 'eeglab2025.0.0');
    if exist(eeglab_path, 'dir')
        addpath(eeglab_path);
    else
        error('EEGLAB not found. Please add it to your MATLAB path.');
    end
end
eeglab nogui;

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
if ~exist('diagnostics/viz_vla_vrw', 'dir')
    mkdir('diagnostics/viz_vla_vrw');
end

%% Display configuration
fprintf('\n');
fprintf('========================================================================\n');
fprintf('  SEED-VLA/VRW DROWSINESS DETECTION PIPELINE\n');
fprintf('========================================================================\n');
fprintf('Labeling Option: %d\n', labeling_option);
switch labeling_option
    case 1
        fprintf('  -> Original thresholds (Alert<0.35, Drowsy>0.70)\n');
        alert_thresh = 0.35;
        drowsy_thresh = 0.70;
    case 2
        fprintf('  -> Relaxed thresholds (Alert<0.40, Drowsy>0.60)\n');
        alert_thresh = 0.40;
        drowsy_thresh = 0.60;
    case 3
        fprintf('  -> Median split (per-subject adaptive)\n');
        alert_thresh = NaN;  % Will be computed per subject
        drowsy_thresh = NaN;
    case 4
        fprintf('  -> Tertile-based (bottom/top 33%%)\n');
        alert_thresh = NaN;  % Will be computed per subject
        drowsy_thresh = NaN;
end
fprintf('Minimum samples per class: %d\n', min_samples_per_class);
fprintf('Segment duration: %d seconds\n', segment_duration_sec);
fprintf('\n');

%% Process all subjects
ALL_X_data = {};
ALL_Y_data = {};
ALL_subject_ids = {};
ALL_perclos_values = {};  % Store for regression option

subject_counter = 0;
subject_info = {};

% Helper function to fix channel names
fix_channel_name = @(name) regexprep(name, '^EEG\s*', '');  % Remove "EEG " prefix
fix_channel_name = @(name) regexprep(regexprep(name, '^EEG\s*', ''), '\s*-\s*Pz$', '');  % Remove " - Pz" suffix

%% Process VLA (Lab) Dataset
fprintf('\n--- Processing VLA (Lab) Dataset ---\n');
vla_files = dir(fullfile(vla_eeg_path, '*.edf'));

for i = 1:length(vla_files)
    orig_subj_id = strrep(vla_files(i).name, '.edf', '');
    eeg_file = fullfile(vla_eeg_path, vla_files(i).name);
    perclos_file = fullfile(vla_perclos_path, [orig_subj_id '.mat']);
    
    fprintf('\nProcessing VLA Subject %s...\n', orig_subj_id);
    
    if ~exist(perclos_file, 'file')
        fprintf('  PERCLOS file not found, skipping.\n');
        continue;
    end
    
    % Load and preprocess EEG
    try
        [X, Y, perclos_vals, alert_count, drowsy_count] = process_vla_vrw_subject(...
            eeg_file, perclos_file, ...
            low_cutoff_freq, high_cutoff_freq, downsample_rate, ...
            segment_duration_sec, skip_initial_sec, ...
            labeling_option, alert_thresh, drowsy_thresh, min_samples_per_class, ...
            standard_channels, ref_channels, frontal_channels, ...
            enable_visualization && (i <= viz_subject_limit), ...
            sprintf('VLA_%s', orig_subj_id));
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        continue;
    end
    
    if isempty(X)
        fprintf('  Subject does not meet criteria (Alert=%d, Drowsy=%d)\n', alert_count, drowsy_count);
        continue;
    end
    
    subject_counter = subject_counter + 1;
    subj_id_str = sprintf('VLA_%s', orig_subj_id);
    
    fprintf('  -> Subject %d (%s): %d samples (Alert=%d, Drowsy=%d)\n', ...
        subject_counter, subj_id_str, length(Y), alert_count, drowsy_count);
    
    % Accumulate data
    subject_ids_for_file = repmat({num2str(subject_counter)}, length(Y), 1);
    ALL_X_data = [ALL_X_data; X];
    ALL_Y_data = [ALL_Y_data; Y];
    ALL_subject_ids = [ALL_subject_ids; subject_ids_for_file];
    ALL_perclos_values = [ALL_perclos_values; num2cell(perclos_vals)];
    
    subject_info{subject_counter} = struct('id', subject_counter, 'name', subj_id_str, ...
        'dataset', 'VLA', 'orig_id', orig_subj_id, 'n_samples', length(Y));
end

%% Process VRW (Real-World) Dataset
fprintf('\n--- Processing VRW (Real-World) Dataset ---\n');
vrw_files = dir(fullfile(vrw_eeg_path, '*.edf'));

for i = 1:length(vrw_files)
    orig_subj_id = strrep(vrw_files(i).name, '.edf', '');
    eeg_file = fullfile(vrw_eeg_path, vrw_files(i).name);
    perclos_file = fullfile(vrw_perclos_path, [orig_subj_id '.mat']);
    
    fprintf('\nProcessing VRW Subject %s...\n', orig_subj_id);
    
    if ~exist(perclos_file, 'file')
        fprintf('  PERCLOS file not found, skipping.\n');
        continue;
    end
    
    % Load and preprocess EEG
    try
        [X, Y, perclos_vals, alert_count, drowsy_count] = process_vla_vrw_subject(...
            eeg_file, perclos_file, ...
            low_cutoff_freq, high_cutoff_freq, downsample_rate, ...
            segment_duration_sec, skip_initial_sec, ...
            labeling_option, alert_thresh, drowsy_thresh, min_samples_per_class, ...
            standard_channels, ref_channels, frontal_channels, ...
            enable_visualization && (i <= viz_subject_limit), ...
            sprintf('VRW_%s', orig_subj_id));
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        continue;
    end
    
    if isempty(X)
        fprintf('  Subject does not meet criteria (Alert=%d, Drowsy=%d)\n', alert_count, drowsy_count);
        continue;
    end
    
    subject_counter = subject_counter + 1;
    subj_id_str = sprintf('VRW_%s', orig_subj_id);
    
    fprintf('  -> Subject %d (%s): %d samples (Alert=%d, Drowsy=%d)\n', ...
        subject_counter, subj_id_str, length(Y), alert_count, drowsy_count);
    
    % Accumulate data
    subject_ids_for_file = repmat({num2str(subject_counter)}, length(Y), 1);
    ALL_X_data = [ALL_X_data; X];
    ALL_Y_data = [ALL_Y_data; Y];
    ALL_subject_ids = [ALL_subject_ids; subject_ids_for_file];
    ALL_perclos_values = [ALL_perclos_values; num2cell(perclos_vals)];
    
    subject_info{subject_counter} = struct('id', subject_counter, 'name', subj_id_str, ...
        'dataset', 'VRW', 'orig_id', orig_subj_id, 'n_samples', length(Y));
end

%% Summary
fprintf('\n========================================================================\n');
fprintf('  PREPROCESSING COMPLETE\n');
fprintf('========================================================================\n');
fprintf('Total valid subjects: %d\n', subject_counter);
fprintf('Total samples: %d\n', length(ALL_Y_data));

if isempty(ALL_Y_data)
    error('No valid subjects found! Try relaxing thresholds or using a different labeling option.');
end

% Convert labels
if iscell(ALL_Y_data)
    ALL_Y_data = string(ALL_Y_data);
end

normal_count = sum(strcmp(ALL_Y_data, 'Alert'));
fatigued_count = sum(strcmp(ALL_Y_data, 'Drowsy'));
fprintf('Label distribution: Alert=%d, Drowsy=%d\n', normal_count, fatigued_count);

%% Create LOSO folds and export
fprintf('\n--- Creating LOSO Cross-Validation Folds ---\n');

unique_subjects = unique(ALL_subject_ids);
k_folds = length(unique_subjects);
fprintf('Number of folds: %d\n', k_folds);

for fold = 1:k_folds
    fprintf('\nFold %d/%d (Test Subject: %s)\n', fold, k_folds, unique_subjects{fold});
    
    test_mask = strcmp(ALL_subject_ids, unique_subjects{fold});
    train_mask = ~test_mask;
    
    % Get indices
    test_indices = find(test_mask);
    train_indices_all = find(train_mask);
    
    % Split train into train/val (80/20)
    n_train = length(train_indices_all);
    perm = randperm(n_train);
    n_val = floor(0.2 * n_train);
    val_indices = train_indices_all(perm(1:n_val));
    train_indices = train_indices_all(perm(n_val+1:end));
    
    % Extract data
    XTrain = ALL_X_data(train_indices);
    YTrain = categorical(ALL_Y_data(train_indices));
    XValidation = ALL_X_data(val_indices);
    YValidation = categorical(ALL_Y_data(val_indices));
    XTest = ALL_X_data(test_indices);
    YTest = categorical(ALL_Y_data(test_indices));
    
    % Export using step4
    step4_export_for_python(XTrain, YTrain, XValidation, YValidation, XTest, YTest, fold, ...
        train_indices, val_indices, test_indices, ALL_subject_ids);
    
    fprintf('  Train: %d, Val: %d, Test: %d\n', length(train_indices), length(val_indices), length(test_indices));
end

%% Save metadata
metadata = struct();
metadata.labeling_option = labeling_option;
metadata.alert_thresh = alert_thresh;
metadata.drowsy_thresh = drowsy_thresh;
metadata.segment_duration_sec = segment_duration_sec;
metadata.n_subjects = subject_counter;
metadata.n_samples = length(ALL_Y_data);
metadata.subject_info = subject_info;

save(fullfile(output_dir, 'vla_vrw_metadata.mat'), 'metadata');

% Save summary file
fid = fopen(fullfile(output_dir, 'folds.txt'), 'w');
fprintf(fid, 'SEED-VLA+VRW LOSO Folds Summary\n');
fprintf(fid, '================================\n\n');
fprintf(fid, 'Labeling Option: %d\n', labeling_option);
fprintf(fid, 'Total subjects: %d\n', subject_counter);
fprintf(fid, 'Total samples: %d\n', length(ALL_Y_data));
fprintf(fid, 'Alert samples: %d\n', normal_count);
fprintf(fid, 'Drowsy samples: %d\n', fatigued_count);
fprintf(fid, '\nSubject Details:\n');
for i = 1:length(subject_info)
    s = subject_info{i};
    fprintf(fid, '  Subject %d: %s (%s) - %d samples\n', s.id, s.name, s.dataset, s.n_samples);
end
fclose(fid);

fprintf('\n========================================================================\n');
fprintf('  EXPORT COMPLETE\n');
fprintf('========================================================================\n');
fprintf('Output directory: %s\n', output_dir);
fprintf('\nTo train the DANN model, run:\n');
fprintf('  python domain_adversarial_training.py --data_dir %s\n', output_dir);

%% ========================================================================
% HELPER FUNCTION: Process a single VLA/VRW subject
% ========================================================================
function [X_out, Y_out, perclos_out, alert_count, drowsy_count] = process_vla_vrw_subject(...
    eeg_file, perclos_file, ...
    low_cutoff, high_cutoff, downsample_rate, ...
    segment_duration_sec, skip_initial_sec, ...
    labeling_option, alert_thresh, drowsy_thresh, min_samples, ...
    standard_channels, ref_channels, frontal_channels, ...
    visualize, subject_id)

    X_out = {};
    Y_out = {};
    perclos_out = [];
    alert_count = 0;
    drowsy_count = 0;
    
    % Load EDF file
    EEG = pop_biosig(eeg_file, 'importevent', 'off');
    original_srate = EEG.srate;
    
    % Fix channel names: "EEG Fp1 - Pz" -> "Fp1"
    for ch = 1:length(EEG.chanlocs)
        old_name = EEG.chanlocs(ch).labels;
        % Remove "EEG " prefix and " - Pz" suffix
        new_name = regexprep(old_name, '^EEG\s*', '');
        new_name = regexprep(new_name, '\s*-\s*Pz$', '');
        EEG.chanlocs(ch).labels = new_name;
    end
    
    % Remove trigger channel if present
    trigger_idx = find(strcmpi({EEG.chanlocs.labels}, 'Trigger'));
    if ~isempty(trigger_idx)
        EEG = pop_select(EEG, 'nochannel', trigger_idx);
    end
    
    % Load PERCLOS
    perclos_data = load(perclos_file);
    perclos_values = perclos_data.perclos(:);
    n_perclos = length(perclos_values);
    
    % Calculate timing
    total_duration = EEG.pnts / original_srate;
    perclos_interval = total_duration / n_perclos;
    
    % Determine thresholds based on labeling option
    switch labeling_option
        case 1  % Original thresholds
            local_alert_thresh = alert_thresh;
            local_drowsy_thresh = drowsy_thresh;
        case 2  % Relaxed thresholds
            local_alert_thresh = alert_thresh;
            local_drowsy_thresh = drowsy_thresh;
        case 3  % Median split
            local_alert_thresh = median(perclos_values);
            local_drowsy_thresh = median(perclos_values);
        case 4  % Tertile-based
            local_alert_thresh = prctile(perclos_values, 33);
            local_drowsy_thresh = prctile(perclos_values, 67);
    end
    
    % Count samples per class (before preprocessing)
    skip_perclos = ceil(skip_initial_sec / perclos_interval);
    valid_perclos = perclos_values(skip_perclos+1:end);
    alert_count = sum(valid_perclos < local_alert_thresh);
    drowsy_count = sum(valid_perclos > local_drowsy_thresh);
    
    % Check if subject meets criteria
    if labeling_option ~= 3  % Option 3 uses all data
        if alert_count < min_samples || drowsy_count < min_samples
            return;
        end
    end
    
    % --- Apply preprocessing pipeline ---
    
    % 1. Select channels (standard + reference)
    all_labels = {EEG.chanlocs.labels};
    all_labels_upper = upper(all_labels);
    
    % Find available standard channels
    standard_upper = upper(standard_channels);
    ref_upper = upper(ref_channels);
    
    idx_standard = find(ismember(all_labels_upper, standard_upper));
    idx_ref = find(ismember(all_labels_upper, ref_upper));
    desired_idx = unique([idx_standard, idx_ref]);
    
    if length(idx_standard) < 10
        warning('Only %d standard channels found, skipping subject.', length(idx_standard));
        return;
    end
    
    EEG = pop_select(EEG, 'channel', desired_idx);
    
    % 2. Add channel locations
    try
        EEG = pop_chanedit(EEG, 'lookup', 'Standard-10-20-Cap81.ced');
    catch
        try
            EEG = pop_chanedit(EEG, 'lookup', 'standard-10-20.elc');
        catch
            warning('Could not look up channel locations.');
        end
    end
    
    % 3. Bandpass filter
    EEG = pop_eegfiltnew(EEG, 'locutoff', low_cutoff, 'hicutoff', high_cutoff);
    
    % 4. Downsample
    EEG = pop_resample(EEG, downsample_rate);
    
    % 5. Re-reference to linked mastoids (if available)
    ref_indices = find(ismember(upper({EEG.chanlocs.labels}), ref_upper));
    if length(ref_indices) >= 2
        EEG = pop_reref(EEG, ref_indices, 'keepref', 'off');
    else
        % Fall back to average reference
        EEG = pop_reref(EEG, []);
    end
    
    % 6. vEOG regression (blink removal)
    eog_idx1 = find(strcmpi('Fp1', {EEG.chanlocs.labels}));
    eog_idx2 = find(strcmpi('F7', {EEG.chanlocs.labels}));
    
    if ~isempty(eog_idx1) && ~isempty(eog_idx2)
        eeg_data = double(EEG.data);
        vEOG = eeg_data(eog_idx1, :) - eeg_data(eog_idx2, :);
        var_vEOG = var(vEOG);
        
        for ch = 1:EEG.nbchan
            if ch == eog_idx1 || ch == eog_idx2
                continue;
            end
            C = cov(eeg_data(ch, :), vEOG);
            if var_vEOG > 0
                b = C(1, 2) / var_vEOG;
            else
                b = 0;
            end
            eeg_data(ch, :) = eeg_data(ch, :) - (b * vEOG);
        end
        EEG.data = single(eeg_data);
    end
    
    % 7. ICA + ICLabel
    try
        EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');
        EEG = pop_iclabel(EEG, 'default');
        
        if isfield(EEG.etc, 'ic_classification') && isfield(EEG.etc.ic_classification, 'ICLabel')
            brain_prob = EEG.etc.ic_classification.ICLabel.classifications(:,1);
            muscle_prob = EEG.etc.ic_classification.ICLabel.classifications(:,2);
            eye_prob = EEG.etc.ic_classification.ICLabel.classifications(:,3);
            artifact_indices = find(brain_prob < eye_prob | brain_prob < muscle_prob);
            
            if ~isempty(artifact_indices)
                EEG = pop_subcomp(EEG, artifact_indices, 0);
            end
        end
    catch ME
        warning('ICA failed: %s', ME.message);
    end
    
    % 8. Select frontal channels for output
    frontal_upper = upper(frontal_channels);
    idx_frontal = find(ismember(upper({EEG.chanlocs.labels}), frontal_upper));
    if ~isempty(idx_frontal)
        EEG = pop_select(EEG, 'channel', idx_frontal);
    end
    
    % --- Extract PERCLOS-labeled segments ---
    segment_samples = segment_duration_sec * downsample_rate;
    skip_samples = skip_initial_sec * downsample_rate;
    
    X_all = {};
    Y_all = {};
    perclos_all = [];
    
    for i = (skip_perclos + 1):n_perclos
        perclos_val = perclos_values(i);
        
        % Determine label
        if perclos_val < local_alert_thresh
            label = 'Alert';
        elseif perclos_val > local_drowsy_thresh
            label = 'Drowsy';
        else
            continue;  % Skip middle range (except for option 3)
        end
        
        % Calculate segment position (3s before PERCLOS evaluation)
        segment_end_time = i * perclos_interval;
        segment_end_sample = round(segment_end_time * downsample_rate);
        segment_start_sample = segment_end_sample - segment_samples + 1;
        
        % Check bounds
        if segment_start_sample < skip_samples || segment_end_sample > EEG.pnts
            continue;
        end
        
        % Extract segment
        data_segment = double(EEG.data(:, segment_start_sample:segment_end_sample));
        
        % Convert to frequency domain (PSD)
        targetLen = 1024;
        if size(data_segment, 2) ~= targetLen
            data_segment = resample(data_segment', targetLen, size(data_segment, 2))';
        end
        spec = abs(fft(data_segment, targetLen, 2));
        spec = spec(:, 1:128);  % Keep first 128 frequency bins
        img = permute(spec, [2 1]);  % freq x channels
        img = reshape(img, [size(img, 1), size(img, 2), 1]);
        
        X_all{end+1, 1} = img;
        Y_all{end+1, 1} = label;
        perclos_all(end+1, 1) = perclos_val;
    end
    
    % Balance classes
    labels_arr = string(Y_all);
    n_alert = sum(labels_arr == "Alert");
    n_drowsy = sum(labels_arr == "Drowsy");
    
    if labeling_option ~= 3  % Balance for options 1, 2, 4
        min_class = min(n_alert, n_drowsy);
        
        alert_idx = find(labels_arr == "Alert");
        drowsy_idx = find(labels_arr == "Drowsy");
        
        rng(42);  % For reproducibility
        selected_alert = alert_idx(randperm(length(alert_idx), min_class));
        selected_drowsy = drowsy_idx(randperm(length(drowsy_idx), min_class));
        
        selected_idx = sort([selected_alert; selected_drowsy]);
        
        X_out = X_all(selected_idx);
        Y_out = Y_all(selected_idx);
        perclos_out = perclos_all(selected_idx);
    else
        X_out = X_all;
        Y_out = Y_all;
        perclos_out = perclos_all;
    end
    
    % Update counts
    labels_final = string(Y_out);
    alert_count = sum(labels_final == "Alert");
    drowsy_count = sum(labels_final == "Drowsy");
end

