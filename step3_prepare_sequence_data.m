function [X_data, Y_data] = step3_prepare_sequence_data(EEG, label, window_length_sec, stride_sec)
% step3_prepare_sequence_data() - Segments continuous data into windows
% of raw time-series data suitable for deep learning.
%
% Usage:
%   >> [X, Y] = step3_prepare_sequence_data(EEG, 'Normal', 5, 1);
%
% Inputs:
%   EEG               - Fully cleaned EEGLAB dataset structure.
%   label             - The session-level label ('Normal' or 'Fatigued').
%   window_length_sec - The duration of each window in seconds.
%   stride_sec        - The stride between window starts in seconds (overlap if < window_length_sec).
%
% Outputs:
%   X_data            - Cell array, where each cell is a [channels x timepoints] matrix.
%   Y_data            - Cell array of corresponding string labels.
%

X_data = {};
Y_data = {};

% Default stride to 1 second if not provided
if nargin < 4 || isempty(stride_sec)
    stride_sec = 1;
end

% --- Calculate window parameters in data points ---
window_length_pnts = floor(window_length_sec * EEG.srate);
% Overlapping windows controlled by stride_sec
step_size_pnts = max(1, floor(stride_sec * EEG.srate));

% --- Start the first window at the beginning of the data ---
current_pos = 1;
window_count = 0;

% --- Loop through the data, sliding the window ---
while (current_pos + window_length_pnts - 1) <= EEG.pnts
    window_count = window_count + 1;
    
    % Define the segment of data for the current window
    window_end = current_pos + window_length_pnts - 1;
    data_segment = EEG.data(:, current_pos:window_end);
    
    %% --- Convert to Power Spectral Density (PSD) image (freq × channels × 1) ---
    % Using PSD (squared magnitude, normalized) to match literature approach
    % Reference: "An EEG-Based Transfer Learning Method for Cross-Subject 
    % Fatigue Mental State Prediction" (Sensors 2021)
    
    targetLen = 1024;
    data_segment = double(data_segment); % ensure double precision
    if size(data_segment,2) ~= targetLen
        data_segment = resample(data_segment', targetLen, size(data_segment,2))'; % channels × 1024
    end
    
    % Compute PSD: |FFT|^2 / N (power spectral density)
    fft_result = fft(data_segment, targetLen, 2);           % channels × 1024
    psd = (abs(fft_result).^2) / targetLen;                 % Power: |FFT|^2 / N
    
    % Keep one-sided spectrum (positive frequencies only)
    % For real signals, PSD is symmetric, so we keep first half and double it
    n_freqs = floor(targetLen/2) + 1;                       % 513 bins for 1024-point FFT
    psd = psd(:, 1:n_freqs);                                % channels × 513
    psd(:, 2:end-1) = 2 * psd(:, 2:end-1);                  % Double non-DC, non-Nyquist bins
    
    % Keep frequencies up to ~30 Hz (relevant for drowsiness: delta, theta, alpha, beta)
    % Frequency resolution = fs / targetLen
    % For fs=250 Hz: each bin = 250/1024 ≈ 0.244 Hz, 30 Hz ≈ 123 bins
    % For fs=128 Hz: each bin = 128/1024 = 0.125 Hz, 30 Hz ≈ 241 bins
    freq_resolution = EEG.srate / targetLen;
    max_freq_hz = 30;
    freq_bins_to_keep = min(n_freqs, ceil(max_freq_hz / freq_resolution) + 1);
    psd = psd(:, 1:freq_bins_to_keep);                      % channels × freq_bins
    
    % Print info on first window only
    if window_count == 1
        fprintf('  PSD: fs=%d Hz, freq_res=%.3f Hz/bin, keeping %d bins (0-%.1f Hz)\n', ...
            EEG.srate, freq_resolution, freq_bins_to_keep, max_freq_hz);
    end
    
    % Rearrange to freq × channels × 1
    img = permute(psd, [2 1]);                              % freq_bins × channels
    img = reshape(img, [size(img,1), size(img,2), 1]);
    
    % Store the image and its label
    X_data{window_count, 1} = img;
    Y_data{window_count, 1} = label;
    
    % Move the window start position forward
    current_pos = current_pos + step_size_pnts;
end

end 