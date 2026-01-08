function EEG_out = step2_run_ica_and_iclabel(EEG_in, frontal_channels, visualize, subject_id)
% step2_run_ica_and_iclabel() - Runs ICA on ALL channels, removes artifacts, then selects frontal channels.
%
% This function:
%   1. Runs ICA on all channels (better decomposition with more channels)
%   2. Uses ICLabel to classify and reject artifact components
%   3. Selects only the desired frontal channels for output
%
% Usage:
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in, frontal_channels);
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in, frontal_channels, true, 'Subject01');
%
% Inputs:
%   EEG_in           - Input EEGLAB dataset structure (preprocessed, all channels).
%   frontal_channels - Cell array of channel labels to keep after ICA (e.g., {'FP1', 'FP2', 'F3', 'FZ', 'F4'}).
%   visualize        - Optional boolean to enable visualization (default: false).
%   subject_id       - Optional subject identifier for plot filenames (default: 'unknown').
%
% Outputs:
%   EEG_out          - EEG dataset with artifacts removed AND only frontal channels.
%
% Requires: ICLabel plugin for EEGLAB.

% Set defaults
if nargin < 3
    visualize = false;
end
if nargin < 4 || isempty(subject_id)
    subject_id = 'unknown';
end

EEG = EEG_in;
fprintf('\n=== Step 2: ICA + ICLabel + Channel Selection ===\n');
fprintf('Input: %d channels, %d data points\n', EEG.nbchan, EEG.pnts);

% Visualize before ICA
if visualize
    viz_path = sprintf('diagnostics/viz/%s_05_before_ica.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 5. Before ICA (%d ch)', subject_id, EEG.nbchan), [], [], viz_path);
end

% --- 1. Check Data Requirements for ICA ---
% ICA needs sufficient data: at least k * channels^2 data points (k=20-30)
min_data_points = 20 * EEG.nbchan^2;
if EEG.pnts < min_data_points
    warning('Data may be insufficient for reliable ICA. Have %d points, recommend at least %d.', ...
        EEG.pnts, min_data_points);
end

% Check data rank (important after average reference)
data_rank = rank(double(EEG.data(:, 1:min(EEG.pnts, 10000))'));  % Sample for speed
fprintf('Data rank: %d (channels: %d)\n', data_rank, EEG.nbchan);

% --- 2. Run ICA on ALL Channels ---
fprintf('Running ICA (extended infomax) on %d channels...\n', EEG.nbchan);
try
    % Use PCA to reduce to data rank if needed (handles rank deficiency from avg ref)
    if data_rank < EEG.nbchan
        fprintf('  Using PCA reduction to rank %d (data is rank-deficient).\n', data_rank);
        EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on', ...
            'pca', data_rank);
    else
        EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');
    end
catch ME
    warning('ICA failed: %s. Skipping ICA/ICLabel, proceeding to channel selection.', ME.message);
    % Skip to channel selection
    EEG_out = select_channels(EEG, frontal_channels, visualize, subject_id);
    return;
end
EEG.comments = pop_comments(EEG.comments, '', ...
    sprintf('Ran extended infomax ICA on %d channels.', EEG.nbchan), 1);
fprintf('  ICA completed. Extracted %d independent components.\n', size(EEG.icaweights, 1));

% --- 3. Run ICLabel ---
fprintf('Running ICLabel for component classification...\n');
if ~exist('pop_iclabel', 'file')
    warning('ICLabel plugin not found. Skipping IC rejection.');
    EEG_out = select_channels(EEG, frontal_channels, visualize, subject_id);
    return;
end

% Ensure required fields exist
if ~isfield(EEG, 'dipfit'), EEG.dipfit = []; end
if ~isfield(EEG, 'icaact'), EEG.icaact = []; end

try
    EEG = pop_iclabel(EEG, 'default');
catch ME
    warning('ICLabel failed: %s. Skipping IC rejection.', ME.message);
    EEG_out = select_channels(EEG, frontal_channels, visualize, subject_id);
    return;
end
EEG.comments = pop_comments(EEG.comments, '', 'Ran ICLabel for component classification.', 1);

% --- 4. Identify and Remove Artifactual Components ---
% Strategy 2 (Moderate): Reject components where:
%   - Brain probability < 30%, OR
%   - Any artifact class probability > 70%
%
% ICLabel classes: 1=Brain, 2=Muscle, 3=Eye, 4=Heart, 5=Line Noise, 6=Channel Noise, 7=Other

if ~isfield(EEG, 'etc') || ~isfield(EEG.etc, 'ic_classification') || ...
   ~isfield(EEG.etc.ic_classification, 'ICLabel')
    warning('ICLabel classification not found. Skipping IC rejection.');
    EEG_out = select_channels(EEG, frontal_channels, visualize, subject_id);
    return;
end

% Get classification probabilities
classifications = EEG.etc.ic_classification.ICLabel.classifications;
brain_prob = classifications(:, 1);
artifact_probs = classifications(:, 2:end);
max_artifact_prob = max(artifact_probs, [], 2);

% Thresholds
brain_threshold = 0.3;      % Reject if brain < 30%
artifact_threshold = 0.7;   % Reject if any artifact > 70%

% Find artifact components
artifact_indices = find(brain_prob < brain_threshold | max_artifact_prob > artifact_threshold);

% Report classification details
fprintf('\nICLabel Component Classification Summary:\n');
fprintf('  Total ICs: %d\n', size(classifications, 1));
class_names = {'Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Chan Noise', 'Other'};

% Count by category
brain_count = sum(brain_prob >= brain_threshold & max_artifact_prob < artifact_threshold);
artifact_count = length(artifact_indices);
fprintf('  Brain components (kept): %d\n', brain_count);
fprintf('  Artifact components (rejected): %d\n', artifact_count);

% Detailed per-component report
fprintf('\n  Per-component breakdown:\n');
for ic = 1:size(classifications, 1)
    [max_prob, max_class] = max(classifications(ic, :));
    is_rejected = ismember(ic, artifact_indices);
    status = '';
    if is_rejected
        status = ' [REJECT]';
    end
    fprintf('    IC%02d: %-10s (%.1f%%), Brain=%.1f%%%s\n', ...
        ic, class_names{max_class}, max_prob*100, brain_prob(ic)*100, status);
end

% Remove artifact components
if ~isempty(artifact_indices)
    fprintf('\nRemoving %d artifact components...\n', length(artifact_indices));
    EEG = pop_subcomp(EEG, artifact_indices, 0);
    EEG.comments = pop_comments(EEG.comments, '', ...
        sprintf('Removed %d artifact ICs (Brain<%.0f%% or Artifact>%.0f%%).', ...
        length(artifact_indices), brain_threshold*100, artifact_threshold*100), 1);
else
    fprintf('\nNo artifact components met rejection criteria.\n');
end

% Visualize after ICA artifact removal (still all channels)
if visualize
    viz_path = sprintf('diagnostics/viz/%s_06_after_ica.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 6. After ICA Artifact Removal (%d ch)', subject_id, EEG.nbchan), [], [], viz_path);
end

% --- 5. Select Frontal Channels ---
EEG_out = select_channels(EEG, frontal_channels, visualize, subject_id);

end


%% Helper Function: Channel Selection
function EEG_out = select_channels(EEG, frontal_channels, visualize, subject_id)
% Select only the specified frontal channels

fprintf('\n--- Channel Selection ---\n');
fprintf('Selecting channels: %s\n', strjoin(frontal_channels, ', '));

all_labels = {EEG.chanlocs.labels};
all_labels_upper = upper(all_labels);
frontal_upper = upper(frontal_channels);

% Find matching channels (case-insensitive)
idx_frontal = find(ismember(all_labels_upper, frontal_upper));

if isempty(idx_frontal)
    warning('None of the requested frontal channels found! Available: %s', strjoin(all_labels, ', '));
    EEG_out = EEG;
    return;
end

% Report which channels were found
found_labels = all_labels(idx_frontal);
missing_labels = setdiff(frontal_upper, all_labels_upper);
fprintf('  Found %d/%d channels: %s\n', length(idx_frontal), length(frontal_channels), strjoin(found_labels, ', '));
if ~isempty(missing_labels)
    fprintf('  Missing channels: %s\n', strjoin(missing_labels, ', '));
end

% Select channels
EEG = pop_select(EEG, 'channel', idx_frontal);
EEG.comments = pop_comments(EEG.comments, '', ...
    sprintf('Selected %d frontal channels: %s', length(idx_frontal), strjoin(found_labels, ', ')), 1);

fprintf('  Final channel count: %d\n', EEG.nbchan);

% Visualize final output
if visualize
    viz_path = sprintf('diagnostics/viz/%s_07_final_frontal.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 7. Final Output (%d frontal ch)', subject_id, EEG.nbchan), [], [], viz_path);
end

EEG_out = EEG;

end
