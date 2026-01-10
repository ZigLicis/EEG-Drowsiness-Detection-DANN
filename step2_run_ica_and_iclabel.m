function EEG_out = step2_run_ica_and_iclabel(EEG_in, visualize, subject_id, manual_review)
% step2_run_ica_and_iclabel() - Runs ICA and uses ICLabel to remove artifacts.
%
% This function runs Independent Component Analysis (ICA) and then uses the
% ICLabel plugin to automatically classify and reject components that are
% not brain-related.
%
% Usage:
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in);
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in, true, 'Subject01');
%   >> EEG_out = step2_run_ica_and_iclabel(EEG_in, true, 'Subject01', true);  % Manual review
%
% Inputs:
%   EEG_in        - Input EEGLAB dataset structure (preprocessed).
%   visualize     - Optional boolean to enable visualization (default: false).
%   subject_id    - Optional subject identifier for plot filenames (default: 'unknown').
%   manual_review - Optional boolean to enable manual component selection (default: false).
%                   When true, displays components and prompts for manual selection.
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
if nargin < 4
    manual_review = false;
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

% Get ICLabel classifications
% Columns: Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other
classifications = EEG.etc.ic_classification.ICLabel.classifications;
class_labels = EEG.etc.ic_classification.ICLabel.classes;
brain_prob = classifications(:,1);
muscle_prob = classifications(:,2);
eye_prob = classifications(:,3);
heart_prob = classifications(:,4);
line_noise_prob = classifications(:,5);
chan_noise_prob = classifications(:,6);
other_prob = classifications(:,7);

% Auto-suggested artifacts (brain < eye OR brain < muscle)
auto_artifact_indices = find(brain_prob < eye_prob | brain_prob < muscle_prob);

if manual_review
    % ==================== MANUAL REVIEW MODE ====================
    fprintf('\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('MANUAL COMPONENT REVIEW - Subject: %s\n', subject_id);
    fprintf('%s\n\n', repmat('=', 1, 70));
    
    % Display component summary table
    fprintf('Component Classification Summary:\n');
    fprintf('%s\n', repmat('-', 1, 90));
    fprintf('%-5s | %-7s | %-7s | %-7s | %-7s | %-7s | %-7s | %-7s | %s\n', ...
        'IC#', 'Brain', 'Muscle', 'Eye', 'Heart', 'Line', 'ChanN', 'Other', 'Auto-Flag');
    fprintf('%s\n', repmat('-', 1, 90));
    
    for ic = 1:size(classifications, 1)
        auto_flag = '';
        if ismember(ic, auto_artifact_indices)
            auto_flag = '*** ARTIFACT';
        end
        fprintf('%-5d | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%% | %s\n', ...
            ic, brain_prob(ic)*100, muscle_prob(ic)*100, eye_prob(ic)*100, ...
            heart_prob(ic)*100, line_noise_prob(ic)*100, chan_noise_prob(ic)*100, ...
            other_prob(ic)*100, auto_flag);
    end
    fprintf('%s\n\n', repmat('-', 1, 90));
    
    fprintf('Auto-suggested artifacts: [%s]\n', num2str(auto_artifact_indices'));
    fprintf('Total components: %d\n\n', size(classifications, 1));
    
    % Open component visualization windows
    fprintf('Opening component visualization...\n');
    fprintf('  1. IC Activations (time series) - look for blinks, muscle bursts\n');
    fprintf('  2. Component scalp maps (topographies)\n');
    fprintf('  3. Component properties (spectra, ERPimage)\n\n');
    
    % Compute IC activations if not already computed
    if isempty(EEG.icaact)
        fprintf('Computing IC activations...\n');
        EEG.icaact = eeg_getdatact(EEG, 'component', 1:size(EEG.icaweights,1));
    end
    
    % 1. Plot IC activations as scrolling time series (like raw EEG but for ICs)
    % This is the KEY visualization for seeing blinks, muscle artifacts, etc.
    try
        fprintf('Opening IC activation scroll plot (look for blink/artifact patterns)...\n');
        % Use eegplot to show IC activations - this shows the raw time series of each IC
        % Blinks appear as sharp spikes, muscle as high-frequency bursts
        eegplot(EEG.icaact, 'srate', EEG.srate, 'title', sprintf('%s: IC Activations (Time Series) - Look for Blinks/Artifacts', subject_id), ...
            'eloc_file', [], 'events', EEG.event, 'winlength', 10, ...
            'dispchans', min(size(EEG.icaact,1), 20));
    catch ME
        fprintf('Could not display IC activations scroll: %s\n', ME.message);
    end
    
    % 2. Plot all component scalp maps (topographies)
    try
        fprintf('Opening scalp map topographies...\n');
        pop_topoplot(EEG, 0, 1:min(size(EEG.icaweights,1), 35), sprintf('%s: IC Scalp Maps', subject_id));
    catch
        fprintf('Could not display scalp maps (may need channel locations).\n');
    end
    
    % 3. Plot ICLabel component properties (spectra, ERPimage, dipole if available)
    try
        fprintf('Opening ICLabel component properties...\n');
        pop_viewprops(EEG, 0, 1:min(size(EEG.icaweights,1), 20), {'freqrange', [1 45]});
    catch
        fprintf('Could not open viewprops (ICLabel visualization).\n');
    end
    
    fprintf('\n');
    fprintf('TIP: In the IC Activations window:\n');
    fprintf('  - Eye blinks appear as LARGE SHARP SPIKES (usually in IC1 or IC2)\n');
    fprintf('  - Muscle artifacts appear as HIGH-FREQUENCY BURSTS\n');
    fprintf('  - Heart artifacts appear as REGULAR RHYTHMIC PATTERNS\n');
    fprintf('  - Line noise appears as CONSTANT 50/60 Hz oscillation\n');
    fprintf('  - Brain activity appears as SMOOTH, IRREGULAR waves\n\n');
    
    % Prompt for manual selection
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('COMPONENT SELECTION\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Review the component plots and enter your selection.\n\n');
    fprintf('Options:\n');
    fprintf('  - Enter component numbers to REJECT (e.g., "1 3 5 7" or "1,3,5,7")\n');
    fprintf('  - Press ENTER with no input to use auto-suggested: [%s]\n', num2str(auto_artifact_indices'));
    fprintf('  - Enter "none" to reject NO components\n');
    fprintf('  - Enter "all" to see component properties one-by-one\n\n');
    
    user_input = input('Components to reject: ', 's');
    
    if isempty(user_input)
        % Use auto-suggested
        artifact_indices = auto_artifact_indices;
        fprintf('Using auto-suggested artifacts: [%s]\n', num2str(artifact_indices'));
    elseif strcmpi(strtrim(user_input), 'none')
        artifact_indices = [];
        fprintf('No components will be rejected.\n');
    elseif strcmpi(strtrim(user_input), 'all')
        % Show each component one by one with time series
        fprintf('\nShowing each component individually with time series...\n');
        fprintf('Each component will show:\n');
        fprintf('  - Scalp map, spectrum, and ERPimage (pop_prop window)\n');
        fprintf('  - Raw IC activation time series (separate figure)\n\n');
        
        reject_list = [];
        for ic = 1:size(EEG.icaweights, 1)
            try
                % Show component properties (scalp map, spectrum, ERPimage)
                pop_prop(EEG, 0, ic, NaN, {'freqrange', [1 45]});
                prop_fig = gcf;
                
                % Also show the IC activation time series for this component
                % This lets you see the actual waveform (blinks, muscle, etc.)
                ts_fig = figure('Name', sprintf('IC %d Time Series', ic), 'NumberTitle', 'off', ...
                    'Position', [100, 100, 1200, 300]);
                
                % Get 30 seconds of data (or all if shorter)
                show_samples = min(30 * EEG.srate, size(EEG.icaact, 2));
                time_vec = (0:show_samples-1) / EEG.srate;
                
                plot(time_vec, EEG.icaact(ic, 1:show_samples), 'k', 'LineWidth', 0.5);
                xlabel('Time (s)');
                ylabel('Amplitude');
                title(sprintf('IC %d Activation (first 30s) - Look for blinks/artifacts', ic), 'FontSize', 12);
                grid on;
                
                % Add artifact type annotation
                [~, max_class] = max(classifications(ic, :));
                class_name = class_labels{max_class};
                annotation('textbox', [0.02, 0.85, 0.3, 0.1], 'String', ...
                    sprintf('ICLabel: %s (%.1f%%)', class_name, classifications(ic, max_class)*100), ...
                    'FitBoxToText', 'on', 'BackgroundColor', 'yellow', 'FontSize', 10);
                
                fprintf('\n--- Component %d ---\n', ic);
                fprintf('ICLabel Classification:\n');
                fprintf('  Brain: %.1f%% | Muscle: %.1f%% | Eye: %.1f%% | Heart: %.1f%%\n', ...
                    brain_prob(ic)*100, muscle_prob(ic)*100, eye_prob(ic)*100, heart_prob(ic)*100);
                fprintf('  Line: %.1f%% | ChanNoise: %.1f%% | Other: %.1f%%\n', ...
                    line_noise_prob(ic)*100, chan_noise_prob(ic)*100, other_prob(ic)*100);
                
                if ismember(ic, auto_artifact_indices)
                    fprintf('  >>> AUTO-FLAGGED as artifact <<<\n');
                end
                
                response = input(sprintf('Reject IC %d? (y/n/q to quit): ', ic), 's');
                if strcmpi(strtrim(response), 'y')
                    reject_list = [reject_list, ic];
                    fprintf('  -> MARKED FOR REJECTION\n');
                elseif strcmpi(strtrim(response), 'q')
                    close(prop_fig);
                    close(ts_fig);
                    break;
                end
                close(prop_fig);
                close(ts_fig);
            catch ME
                fprintf('Could not display component %d: %s\n', ic, ME.message);
            end
        end
        artifact_indices = reject_list;
        fprintf('\nSelected for rejection: [%s]\n', num2str(artifact_indices));
    else
        % Parse user input
        user_input = strrep(user_input, ',', ' ');
        artifact_indices = str2num(user_input); %#ok<ST2NM>
        if isempty(artifact_indices)
            fprintf('Invalid input. Using auto-suggested artifacts.\n');
            artifact_indices = auto_artifact_indices;
        else
            % Validate indices
            valid_range = 1:size(EEG.icaweights, 1);
            artifact_indices = artifact_indices(ismember(artifact_indices, valid_range));
            fprintf('Will reject components: [%s]\n', num2str(artifact_indices));
        end
    end
    
    % Close any remaining figures from visualization
    fprintf('\nClose the component visualization windows when ready, then press ENTER to continue...\n');
    pause;
    
else
    % ==================== AUTOMATIC MODE ====================
    artifact_indices = auto_artifact_indices;
end

% Remove selected components
if ~isempty(artifact_indices)
    fprintf('Removing %d artifact components: [%s]\n', length(artifact_indices), num2str(artifact_indices'));
    EEG = pop_subcomp(EEG, artifact_indices, 0);
    EEG.comments = pop_comments(EEG.comments, '', sprintf('Removed %d artifact ICs: %s', length(artifact_indices), num2str(artifact_indices')), 1);
else
    fprintf('No components removed.\n');
end

% Visualize after ICA
if visualize
    viz_path = sprintf('diagnostics/viz/%s_06_after_ica.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 6. After ICA/ICLabel Artifact Removal', subject_id), [], [], viz_path);
end

EEG_out = EEG;

end 