%% MAIN DROWSINESS ANALYSIS PIPELINE (DEEP LEARNING)
% This script implements an end-to-end deep learning pipeline to detect
% driver drowsiness from raw EEG recordings (.cnt files).
%
% Pipeline Steps:
% 1. Data Traversal: Scans a root directory for subject folders and their
%    corresponding 'Normal' and 'Fatigued' state EEG files.
% 2. Preprocessing (Step 1): Full preprocessing on ALL channels:
%    - Channel location lookup
%    - Remove non-EEG channels
%    - Bandpass filtering (1-45 Hz)
%    - Downsampling (to target rate)
%    - Bad channel detection & interpolation
%    - ASR (Artifact Subspace Reconstruction) for large transients
%    - Re-reference to average
% 3. Artifact Removal (Step 2): ICA on ALL channels + ICLabel + Channel Selection
%    - Extended Infomax ICA decomposition (on all channels for better separation)
%    - ICLabel classification (Brain vs. artifacts)
%    - Removes artifact components (Strategy 2: Brain<30% or Artifact>70%)
%    - Selects frontal channels AFTER artifact removal
% 4. Data Preparation (Step 3): Segments continuous data into frequency-domain windows
% 5. Data Partitioning: Leave-One-Subject-Out (LOSO) cross-validation split
% 6. Export (Step 4): Export to Python for deep learning training
%
% To Use:
% 1. Update the 'data_root_path' variable below.
% 2. Ensure EEGLAB and plugins are installed (ICLabel, clean_rawdata).
% 3. Run this script from the MATLAB command window or editor.
%
% Requires: EEGLAB, ICLabel plugin, clean_rawdata plugin (for ASR),
%           Signal Processing Toolbox, Statistics and Machine Learning Toolbox.
%

% --- USER-DEFINED PARAMETERS ---
% Path to The Original EEG Data for Driver Fatigue Detection
data_root_path = '/Users/ziglicis/Desktop/Research/ResearchDatasets/TheOriginalEEG';

% Preprocessing Parameters (passed to step1)
low_cutoff_freq  = 1;    % 1-45 Hz bandpass filter
high_cutoff_freq = 45;
downsample_rate  = 250;  % Downsampled to 250 Hz 

% Channel Selection (applied AFTER ICA in step2)
frontal_channels = {'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8'};

% Windowing Parameters (passed to step3)
window_length_sec = 5;   % 5-second windows
stride_seconds = 1;      % overlapping windows every 1 second

% Visualization Parameters
enable_visualization = true;  % Set to true to generate before/after plots
viz_subject_limit = 12;        % Only visualize first N subjects (set to Inf for all)

% --- END OF PARAMETERS ---

% Start EEGLAB if not running
if ~exist('eeglab', 'file')
    % Try to find EEGLAB in the parent directory
    eeglab_path = fullfile(fileparts(pwd), 'eeglab2025.0.0');
    if exist(eeglab_path, 'dir')
        fprintf('Adding EEGLAB to path: %s\n', eeglab_path);
        addpath(eeglab_path);
    else
        error('EEGLAB not found. Please add it to your MATLAB path or ensure eeglab2025.0.0 folder exists in parent directory.');
    end
end

% Properly initialize EEGLAB (this loads plugins)
fprintf('Initializing EEGLAB...\n');
eeglab nogui; % Start EEGLAB without GUI to load all plugins

% Check if required loading functions are available
if ~exist('pop_loadcnt', 'file')
    error('pop_loadcnt function not found. Make sure the Neuroscan plugin is installed and EEGLAB is properly initialized.');
end

% Check for recommended plugins
if ~exist('clean_asr', 'file')
    warning('clean_rawdata plugin not found. ASR will be skipped. Install via EEGLAB > File > Manage Extensions.');
end
if ~exist('pop_iclabel', 'file')
    warning('ICLabel plugin not found. IC classification will fail. Install via EEGLAB > File > Manage Extensions.');
end

% Create output directories
if ~exist('diagnostics/viz', 'dir')
    mkdir('diagnostics/viz');
end

% --- 1. Data Traversal and Preparation ---
fprintf('\n');
fprintf('========================================================================\n');
fprintf('  DROWSINESS DETECTION PIPELINE - ICA on All Channels + ASR\n');
fprintf('========================================================================\n');
fprintf('--- Phase 1: Data Traversal and Preparation ---\n');

ALL_X_data = {};
ALL_Y_data = {};
ALL_subject_ids = {}; % Track which subject each window came from

subject_folders = dir(fullfile(data_root_path, '*'));
subject_folders = subject_folders([subject_folders.isdir] & ~ismember({subject_folders.name},{'.','..'}));

fprintf('Found %d subject folders:\n', length(subject_folders));
for i = 1:length(subject_folders)
    fprintf('  %s\n', subject_folders(i).name);
end
fprintf('\n');

% Loop through each subject folder
for i = 1:length(subject_folders)
    subject_id = subject_folders(i).name;
    subject_path = fullfile(data_root_path, subject_id);
    fprintf('\n========================================================================\n');
    fprintf('  Processing Subject %d/%d: %s\n', i, length(subject_folders), subject_id);
    fprintf('========================================================================\n');
    
    % Define .cnt files to process
    files_to_process = {
        struct('path', fullfile(subject_path, 'Normal state.cnt'), 'label', 'Normal'),...
        struct('path', fullfile(subject_path, 'Fatigue state.cnt'), 'label', 'Fatigued')
    };
    
    for j = 1:length(files_to_process)
        file_info = files_to_process{j};
        fprintf('\n--- Processing: %s ---\n', file_info.label);
        
        if ~exist(file_info.path, 'file')
            fprintf('  File not found, skipping: %s\n', file_info.path);
            continue; 
        end
        
        fprintf('Loading file: %s\n', file_info.path);
        
        % Load .cnt file
        try
            try
                EEG = pop_loadcnt(file_info.path, 'dataformat', 'auto');
            catch ME1
                fprintf('  pop_loadcnt failed, trying pop_biosig...\n');
                EEG = pop_biosig(file_info.path);
            end
            fprintf('  Loaded: %d channels, %d points, %.1f seconds @ %d Hz\n', ...
                EEG.nbchan, EEG.pnts, EEG.pnts/EEG.srate, EEG.srate);
        catch ME
            fprintf('  ✗ File loading failed: %s\n', ME.message);
            continue;
        end
        
        % Determine if visualization should be enabled for this subject
        should_visualize = enable_visualization && (i <= viz_subject_limit);
        
        % STEP 1: PREPROCESSING (filtering, downsampling, ASR, re-reference - on ALL channels)
        try
            EEG = step1_preprocess_data(EEG, low_cutoff_freq, high_cutoff_freq, downsample_rate, ...
                should_visualize, subject_id);
        catch ME
            fprintf('  ✗ Preprocessing failed: %s\n', ME.message);
            fprintf('  Skipping this file.\n');
            continue;
        end
        
        % STEP 2: ICA ON ALL CHANNELS + ICLABEL + CHANNEL SELECTION
        try
            EEG = step2_run_ica_and_iclabel(EEG, frontal_channels, should_visualize, subject_id);
        catch ME
            fprintf('  ✗ ICA/ICLabel failed: %s\n', ME.message);
            fprintf('  Attempting to continue with channel selection only...\n');
            % Fallback: just select channels without ICA
            try
                all_labels = {EEG.chanlocs.labels};
                idx = find(ismember(upper(all_labels), upper(frontal_channels)));
                if ~isempty(idx)
                    EEG = pop_select(EEG, 'channel', idx);
                end
            catch
                fprintf('  ✗ Channel selection also failed. Skipping this file.\n');
                continue;
            end
        end
        
        % STEP 3: Prepare sequence data (windowing)
        [X, Y] = step3_prepare_sequence_data(EEG, file_info.label, window_length_sec, stride_seconds);
        fprintf('  Generated %d windows from this file\n', length(Y));
        
        % Create subject ID array for this file's windows
        subject_ids_for_file = repmat({subject_id}, length(Y), 1);
        
        % Accumulate results
        ALL_X_data = [ALL_X_data; X];
        ALL_Y_data = [ALL_Y_data; Y];
        ALL_subject_ids = [ALL_subject_ids; subject_ids_for_file];
    end
end

fprintf('\n\n========================================================================\n');
fprintf('  All subjects processed. Total windows collected: %d\n', length(ALL_Y_data));
fprintf('========================================================================\n');

% Check if any data was collected
if isempty(ALL_Y_data) || length(ALL_Y_data) == 0
    error('No data windows were collected! Please check:\n1. Data path is correct\n2. Files "Normal state.cnt" and "Fatigue state.cnt" exist in subject folders\n3. Files can be loaded by EEGLAB');
end

% Convert labels to categorical for analysis
if iscell(ALL_Y_data)
    ALL_Y_data = string(ALL_Y_data);
end

% --- 2. Subject-Based Data Splitting ---
fprintf('\n--- Phase 2: Subject-Based Data Partitioning ---\n');

unique_subjects = unique(ALL_subject_ids);
fprintf('Total subjects found: %d\n', length(unique_subjects));
fprintf('Subjects: %s\n', strjoin(unique_subjects, ', '));

% Verify preprocessing
fprintf('\n--- Preprocessing Verification ---\n');
normal_count = sum(strcmp(ALL_Y_data, 'Normal'));
fatigued_count = sum(strcmp(ALL_Y_data, 'Fatigued'));
fprintf('Label distribution: Normal=%d, Fatigued=%d (%.1f%% / %.1f%%)\n', ...
    normal_count, fatigued_count, ...
    100*normal_count/length(ALL_Y_data), 100*fatigued_count/length(ALL_Y_data));

% Check data distribution per subject
fprintf('\nData distribution per subject:\n');
for i = 1:length(unique_subjects)
    subj_indices = strcmp(ALL_subject_ids, unique_subjects{i});
    subj_labels = ALL_Y_data(subj_indices);
    subj_normal = sum(strcmp(subj_labels, 'Normal'));
    subj_fatigued = sum(strcmp(subj_labels, 'Fatigued'));
    fprintf('  %s: %d windows (Normal=%d, Fatigued=%d)\n', ...
        unique_subjects{i}, length(subj_labels), subj_normal, subj_fatigued);
end

% --- 3. Leave-One-Subject-Out (LOSO) Cross-Validation ---
fprintf('\n--- Phase 3: Leave-One-Subject-Out (LOSO) Cross-Validation ---\n');

k_folds = length(unique_subjects);  % One fold per subject (LOSO)
% Randomize subject order for LOSO
rng('default');
rng(42, 'twister'); % For reproducibility
shuffled_subjects = unique_subjects(randperm(k_folds));

fprintf('Performing LOSO cross-validation across %d subjects...\n', k_folds);

cv_accuracies = [];

for fold = 1:k_folds
    fprintf('\n--- Fold %d/%d (Test Subject: %s) ---\n', fold, k_folds, shuffled_subjects{fold});
    % Determine test subject for this fold
    test_subjects = shuffled_subjects(fold);
    train_subjects = setdiff(shuffled_subjects, test_subjects);
    
    fprintf('Test subjects: %s\n', strjoin(test_subjects, ', '));
    fprintf('Train subjects: %s\n', strjoin(train_subjects, ', '));
    
    % Create train/test splits based on subjects
    test_mask = ismember(ALL_subject_ids, test_subjects);
    train_mask = ~test_mask;
    
    % Further split training data into train/validation (80/20)
    train_indices = find(train_mask);
    val_split = cvpartition(length(train_indices), 'HoldOut', 0.2);
    
    final_train_indices = train_indices(val_split.training);
    val_indices = train_indices(val_split.test);
    test_indices = find(test_mask);
    
    % Extract data for this fold
    XTrain = ALL_X_data(final_train_indices);
    YTrain = categorical(ALL_Y_data(final_train_indices));
    XValidation = ALL_X_data(val_indices);
    YValidation = categorical(ALL_Y_data(val_indices));
    XTest = ALL_X_data(test_indices);
    YTest = categorical(ALL_Y_data(test_indices));

    %% --- SUBJECT-WISE SPECTRAL NORMALIZATION ---
    % Normalize each subject's spectral patterns to their own baseline
    % This reduces inter-subject spectral signature differences
    
    % Step 1: Compute per-subject statistics from training data only
    train_subject_stats = containers.Map();
    for subj_idx = 1:length(unique_subjects)
        subj_id = unique_subjects{subj_idx};
        if ismember(subj_id, train_subjects)
            % Find this subject's windows in training set
            subj_train_mask = ismember(ALL_subject_ids(final_train_indices), subj_id);
            if sum(subj_train_mask) > 0
                subj_windows = XTrain(subj_train_mask);
                % Concatenate all spectral images for this subject
                subj_spectra = cat(4, subj_windows{:}); % freq x ch x 1 x windows
                subj_spectra = reshape(subj_spectra, [], size(subj_spectra,4)); % (freq*ch) x windows
                % Compute mean and std across windows for this subject
                subj_mean = mean(subj_spectra, 2);
                subj_std = std(subj_spectra, 0, 2) + eps;
                train_subject_stats(subj_id) = struct('mean', subj_mean, 'std', subj_std);
            end
        end
    end
    
    % Step 2: Apply subject-wise normalization to all sets
    XTrain = apply_subject_normalization(XTrain, ALL_subject_ids(final_train_indices), train_subject_stats);
    XValidation = apply_subject_normalization(XValidation, ALL_subject_ids(val_indices), train_subject_stats);
    
    % For test subject, compute stats from their own data (unsupervised adaptation)
    test_subj_id = test_subjects{1};
    test_windows = XTest;
    test_spectra = cat(4, test_windows{:});
    test_spectra = reshape(test_spectra, [], size(test_spectra,4));
    test_mean = mean(test_spectra, 2);
    test_std = std(test_spectra, 0, 2) + eps;
    test_stats = containers.Map();
    test_stats(test_subj_id) = struct('mean', test_mean, 'std', test_std);
    XTest = apply_subject_normalization(XTest, ALL_subject_ids(test_indices), test_stats);

    % If data are 2-D images (freq×channels) skip manual z-score; rely on imageInputLayer
    if ndims(XTrain{1}) == 2
        %% Manual z-score for sequence data (legacy path)
        concatTrain = cat(2, XTrain{:});
        mu  = mean(concatTrain, 2);
        sig = std(concatTrain, 0, 2) + eps;
        normFn = @(x) (x - mu) ./ sig;
        XTrain      = cellfun(normFn, XTrain,      'UniformOutput', false);
        XValidation = cellfun(normFn, XValidation, 'UniformOutput', false);
        XTest       = cellfun(normFn, XTest,       'UniformOutput', false);
    end
    
    fprintf('  Training samples: %d\n', length(XTrain));
    fprintf('  Validation samples: %d\n', length(XValidation));
    fprintf('  Test samples: %d\n', length(XTest));
    
    %% --- EXPORT DATA FOR PYTHON TRAINING ---
    fprintf('\nExporting data for fold %d...\n', fold);
    try
        % Export data for Python deep learning
        step4_export_for_python(XTrain, YTrain, XValidation, YValidation, XTest, YTest, fold, ...
            final_train_indices, val_indices, test_indices, ALL_subject_ids);
        
        % Placeholder accuracy (will be filled by Python results)
        fold_accuracy = NaN;
        
    catch ME
        fprintf('Error in fold %d: %s\n', fold, ME.message);
        fold_accuracy = NaN;
    end
    
    cv_accuracies = [cv_accuracies, fold_accuracy];
end

%% --- FINAL RESULTS ---
fprintf('\n========================================================================\n');
fprintf('  MATLAB PREPROCESSING COMPLETE\n');
fprintf('========================================================================\n');
fprintf('Data exported for %d folds to diagnostics/python_data/ directory\n', k_folds);
fprintf('\nPipeline Summary:\n');
fprintf('  - Preprocessing: Filter (%.0f-%.0f Hz), Downsample (%d Hz), ASR, Avg Ref\n', ...
    low_cutoff_freq, high_cutoff_freq, downsample_rate);
fprintf('  - ICA: Extended Infomax on ALL channels\n');
fprintf('  - ICLabel: Strategy 2 (Brain<30%% or Artifact>70%%)\n');
fprintf('  - Channel Selection: %s\n', strjoin(frontal_channels, ', '));
fprintf('  - Windowing: %d sec windows, %d sec stride\n', window_length_sec, stride_seconds);
fprintf('\nNext steps:\n');
fprintf('1. Run the Python deep learning script: python step4_domain_adversarial_training.py\n');
fprintf('2. Python will train models with domain adversarial training\n');
fprintf('3. Results will be saved back to MATLAB format\n');

% Save export completion flag
if ~exist('diagnostics/python_data', 'dir')
    mkdir('diagnostics/python_data');
end
save('diagnostics/python_data/export_complete.mat', 'k_folds', 'unique_subjects', ...
    'frontal_channels', 'low_cutoff_freq', 'high_cutoff_freq', 'downsample_rate');

fprintf('\nMATLAB preprocessing completed successfully!\n');
fprintf('\n*** Subject-Based Cross-Validation Pipeline Complete! ***\n');
