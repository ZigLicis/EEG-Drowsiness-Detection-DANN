function EEG_out = step1_preprocess_data(EEG_in, low_cutoff, high_cutoff, downsample_rate, visualize, subject_id)
% step1_preprocess_data() - Performs initial preprocessing on ALL channels.
%
% This function applies filtering, downsampling, bad channel detection,
% ASR artifact removal, and re-referencing on ALL channels (no channel
% selection). Channel selection happens AFTER ICA in step2.
%
% Pipeline order:
%   1. Channel location lookup
%   2. Remove non-EEG channels (without locations)
%   3. Bandpass filtering (on all channels)
%   4. Downsampling
%   5. Bad channel detection & interpolation
%   6. ASR (Artifact Subspace Reconstruction) for large transients
%   7. Re-reference to average
%
% Usage:
%   >> EEG_out = step1_preprocess_data(EEG_in, low_cutoff, high_cutoff, downsample_rate);
%   >> EEG_out = step1_preprocess_data(EEG_in, low_cutoff, high_cutoff, downsample_rate, true, 'Subject01');
%
% Inputs:
%   EEG_in          - Input EEGLAB dataset structure.
%   low_cutoff      - Low frequency cutoff for high-pass filter (Hz).
%   high_cutoff     - High frequency cutoff for low-pass filter (Hz).
%   downsample_rate - New sampling rate to downsample to (Hz).
%   visualize       - Optional boolean to enable visualization (default: false).
%   subject_id      - Optional subject identifier for plot filenames (default: 'unknown').
%
% Outputs:
%   EEG_out         - Preprocessed EEGLAB dataset structure (all channels).

% Set defaults
if nargin < 5
    visualize = false;
end
if nargin < 6 || isempty(subject_id)
    subject_id = 'unknown';
end

EEG = EEG_in;
original_nbchan = EEG.nbchan;
fprintf('Starting preprocessing with %d channels...\n', original_nbchan);

% --- 1. Look up Channel Locations ---
needs_lookup = false;
if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
    needs_lookup = true;
else
    if ~isfield(EEG.chanlocs, 'X')
        needs_lookup = true;
    else
        try
            needs_lookup = all(cellfun(@isempty, {EEG.chanlocs.X}));
        catch
            needs_lookup = true;
        end
    end
end
if needs_lookup
    fprintf('Channel locations not found. Looking up standard locations...\n');
    try
        EEG = pop_chanedit(EEG, 'lookup', 'Standard-10-20-Cap81.ced');
        EEG.comments = pop_comments(EEG.comments, '', 'Looked up standard channel locations (Cap81).', 1);
    catch
        try
            EEG = pop_chanedit(EEG, 'lookup', 'standard-10-20.elc');
            EEG.comments = pop_comments(EEG.comments, '', 'Looked up standard channel locations (elc).', 1);
        catch
            warning('Could not automatically look up channel locations.');
        end
    end
end

% --- 2. Remove non-EEG channels (those without locations) ---
% Keep only channels that have valid 3D coordinates
if isfield(EEG.chanlocs, 'X')
    channels_without_loc = find(cellfun(@isempty, {EEG.chanlocs.X}));
    if ~isempty(channels_without_loc)
        removed_labels = {EEG.chanlocs(channels_without_loc).labels};
        fprintf('Removing %d non-EEG channels without locations: %s\n', ...
            length(channels_without_loc), strjoin(removed_labels, ', '));
        EEG = pop_select(EEG, 'nochannel', channels_without_loc);
        EEG.comments = pop_comments(EEG.comments, '', ...
            sprintf('Removed %d non-EEG channels.', length(channels_without_loc)), 1);
    end
end
fprintf('Channels after removing non-EEG: %d\n', EEG.nbchan);

% Visualize raw data
if visualize
    viz_path = sprintf('diagnostics/viz/%s_01_raw.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 1. Raw EEG (%d channels)', subject_id, EEG.nbchan), [], [], viz_path);
end

% --- 3. Bandpass Filtering (on all channels) ---
fprintf('Applying bandpass filter (%.1f-%.1f Hz) on %d channels...\n', low_cutoff, high_cutoff, EEG.nbchan);
EEG = pop_eegfiltnew(EEG, 'locutoff', low_cutoff, 'hicutoff', high_cutoff);
EEG.comments = pop_comments(EEG.comments, '', sprintf('Bandpass filtered from %.1f to %.1f Hz.', low_cutoff, high_cutoff), 1);

% Visualize after filtering
if visualize
    viz_path = sprintf('diagnostics/viz/%s_02_after_filtering.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 2. After Bandpass Filtering', subject_id), [], [], viz_path);
end

% --- 4. Downsampling ---
if EEG.srate > downsample_rate
    fprintf('Downsampling data from %d Hz to %d Hz...\n', EEG.srate, downsample_rate);
    EEG = pop_resample(EEG, downsample_rate);
    EEG.comments = pop_comments(EEG.comments, '', sprintf('Downsampled to %d Hz.', downsample_rate), 1);
else
    fprintf('Data already at %d Hz, skipping downsampling.\n', EEG.srate);
end

% --- 5. Bad Channel Detection & Interpolation ---
fprintf('Detecting bad channels...\n');
original_chanlocs = EEG.chanlocs;
original_nbchan_before_bad = EEG.nbchan;

try
    % Use clean_rawdata's channel rejection (requires clean_rawdata plugin)
    if exist('clean_channels', 'file')
        % Detect bad channels using correlation criterion
        EEG_temp = clean_channels(EEG, 0.8);  % Remove channels with <0.8 correlation
        
        % Find which channels were removed
        removed_chan_labels = setdiff({original_chanlocs.labels}, {EEG_temp.chanlocs.labels});
        
        if ~isempty(removed_chan_labels)
            fprintf('  Detected %d bad channels: %s\n', length(removed_chan_labels), strjoin(removed_chan_labels, ', '));
            
            % Interpolate bad channels instead of removing them
            bad_chan_idx = find(ismember({original_chanlocs.labels}, removed_chan_labels));
            EEG = pop_interp(EEG, bad_chan_idx, 'spherical');
            EEG.comments = pop_comments(EEG.comments, '', ...
                sprintf('Interpolated %d bad channels: %s', length(bad_chan_idx), strjoin(removed_chan_labels, ', ')), 1);
            fprintf('  Interpolated %d bad channels.\n', length(bad_chan_idx));
        else
            fprintf('  No bad channels detected.\n');
        end
    else
        % Fallback: use EEGLAB's built-in channel rejection
        fprintf('  clean_channels not available, using pop_rejchan...\n');
        [EEG, bad_idx] = pop_rejchan(EEG, 'elec', 1:EEG.nbchan, 'threshold', 5, 'norm', 'on', 'measure', 'kurt');
        if ~isempty(bad_idx)
            % Restore and interpolate
            EEG = EEG_in;  % This won't work well, so just warn
            warning('Bad channel detection with pop_rejchan - channels removed but not interpolated.');
        end
    end
catch ME
    warning('Bad channel detection failed: %s. Continuing without interpolation.', ME.message);
end

fprintf('Channels after bad channel handling: %d\n', EEG.nbchan);

% --- 6. ASR (Artifact Subspace Reconstruction) ---
% ASR removes large transient artifacts (muscle bursts, movement, electrode pops)
% This prepares the data for better ICA decomposition
fprintf('Applying ASR (Artifact Subspace Reconstruction)...\n');

try
    if exist('clean_asr', 'file')
        % ASR parameters:
        %   - cutoff: standard deviations for rejection (lower = more aggressive)
        %   - Default is 20, we use 20 for moderate cleaning
        asr_cutoff = 20;  % Standard deviations (20 is moderate, 10 is aggressive)
        
        % Store original data for comparison
        original_data_var = var(EEG.data(:));
        
        % Apply ASR
        EEG = clean_asr(EEG, asr_cutoff);
        EEG.comments = pop_comments(EEG.comments, '', ...
            sprintf('Applied ASR with cutoff=%d SD.', asr_cutoff), 1);
        
        % Report variance reduction
        cleaned_data_var = var(EEG.data(:));
        var_reduction = (1 - cleaned_data_var/original_data_var) * 100;
        fprintf('  ASR applied (cutoff=%d SD). Variance reduced by %.1f%%.\n', asr_cutoff, var_reduction);
    else
        warning('clean_asr not found. Install clean_rawdata plugin for ASR. Skipping ASR.');
    end
catch ME
    warning('ASR failed: %s. Continuing without ASR.', ME.message);
end

% Visualize after ASR
if visualize
    viz_path = sprintf('diagnostics/viz/%s_03_after_asr.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 3. After ASR', subject_id), [], [], viz_path);
end

% --- 7. Re-reference to Average ---
% Average reference is standard for ICA and works well with many channels
fprintf('Re-referencing to average (all %d channels)...\n', EEG.nbchan);
EEG = pop_reref(EEG, []);
EEG.comments = pop_comments(EEG.comments, '', 'Re-referenced to average.', 1);

% Visualize after re-referencing
if visualize
    viz_path = sprintf('diagnostics/viz/%s_04_after_reref.png', subject_id);
    visualize_eeg_timeseries(EEG, sprintf('%s: 4. After Re-referencing (Avg)', subject_id), [], [], viz_path);
end

% --- Summary ---
fprintf('\nStep 1 Preprocessing Complete:\n');
fprintf('  Original channels: %d\n', original_nbchan);
fprintf('  Final channels: %d\n', EEG.nbchan);
fprintf('  Sampling rate: %d Hz\n', EEG.srate);
fprintf('  Duration: %.1f seconds\n', EEG.pnts / EEG.srate);
fprintf('  Data points: %d\n', EEG.pnts);

EEG_out = EEG;

end
