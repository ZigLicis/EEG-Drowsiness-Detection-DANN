function EEG_out = step2_run_ica_and_iclabel(EEG_in, visualize, subject_id)
% step2_run_ica_and_iclabel() - Runs ICA and uses ICLabel to remove artifacts.
%
% This function runs Independent Component Analysis (ICA) and then uses the
% ICLabel plugin to automatically classify and reject components that are
% not brain-related.
%
% Usage:
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in);
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in, true, 'Subject01');
%
% Inputs:
%   EEG_in      - Input EEGLAB dataset structure (preprocessed).
%   visualize   - Optional boolean to enable visualization (default: false).
%   subject_id  - Optional subject identifier for plot filenames (default: 'unknown').
%
% Outputs:
%   EEG_out     - EEG dataset with artifactual ICs removed.
%
% Requires: ICLabel plugin for EEGLAB.

% Set defaults
if nargin < 2
    visualize = false;
end
if nargin < 3 || isempty(subject_id)
    subject_id = 'unknown';
end

EEG = EEG_in;

% Visualize before ICA
if visualize
    viz_path = sprintf('diagnostics/viz/%s_05_before_ica.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 5. Before ICA', subject_id), [], [], viz_path);
end

% --- 1. Run ICA ---
% Let ICA automatically determine the data rank to handle potential
% rank-deficiency issues after preprocessing, which is more robust than
% manually setting the PCA dimension.
fprintf('Running ICA (infomax)...\n');
try
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');
catch ME
    warning('ICA failed: %s. Skipping ICA/ICLabel for this dataset.', ME.message);
    EEG_out = EEG;
    return;
end
EEG.comments = pop_comments(EEG.comments, '', 'Ran extended infomax ICA.', 1);

% --- 2. Run ICLabel ---
% This automatically classifies components into categories like brain, muscle, eye, etc.
fprintf('Running ICLabel for component classification...\n');
if ~exist('pop_iclabel', 'file')
    error('ICLabel plugin not found. Please install it through the EEGLAB extension manager.');
end
% Ensure fields some plugins expect exist
if ~isfield(EEG, 'dipfit')
    EEG.dipfit = [];
end
if ~isfield(EEG, 'icaact')
    EEG.icaact = [];
end
try
    EEG = pop_iclabel(EEG, 'default');
catch ME
    warning('ICLabel failed: %s. Proceeding without IC rejection.', ME.message);
    EEG_out = EEG;
    return;
end
EEG.comments = pop_comments(EEG.comments, '', 'Ran ICLabel for component classification.', 1);

% --- 3. Identify and Remove Artifactual Components ---
% After vEOG cleaning in Step 1, the remaining artifacts are more subtle.
% We use a more sensitive threshold here: flag a component for rejection if
% its 'Brain' probability is lower than its 'Eye' or 'Muscle' probability.

if ~isfield(EEG,'etc') || ~isfield(EEG.etc,'ic_classification') || ~isfield(EEG.etc.ic_classification,'ICLabel')
    EEG_out = EEG;
    return;
end
brain_prob = EEG.etc.ic_classification.ICLabel.classifications(:,1);
muscle_prob = EEG.etc.ic_classification.ICLabel.classifications(:,2);
eye_prob = EEG.etc.ic_classification.ICLabel.classifications(:,3);

artifact_indices = find(brain_prob < eye_prob | brain_prob < muscle_prob);

if ~isempty(artifact_indices)
    fprintf('Removing %d artifact components identified by ICLabel.\n', length(artifact_indices));
    EEG = pop_subcomp(EEG, artifact_indices, 0);
    EEG.comments = pop_comments(EEG.comments, '', sprintf('Removed %d artifact ICs.', length(artifact_indices)), 1);
else
    fprintf('No significant artifact components found by ICLabel.\n');
end

% Visualize after ICA
if visualize
    viz_path = sprintf('diagnostics/viz/%s_06_after_ica.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 6. After ICA/ICLabel Artifact Removal', subject_id), [], [], viz_path);
end

EEG_out = EEG;

end 