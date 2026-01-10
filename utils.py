"""
Shared utility functions for EEG Drowsiness Detection Pipeline
==============================================================

This module contains common functions used across multiple Python scripts.
"""

import numpy as np
import h5py


def load_matlab_v73(filename):
    """
    Load MATLAB v7.3 files using h5py.
    
    MATLAB v7.3 files use HDF5 format, which requires h5py to read.
    This function handles the conversion from MATLAB to Python data types.
    
    Parameters
    ----------
    filename : str
        Path to the .mat file (v7.3 format)
    
    Returns
    -------
    dict
        Dictionary containing the loaded data with keys matching MATLAB variable names
    
    Example
    -------
    >>> data = load_matlab_v73('fold_1_data.mat')
    >>> X_train = data['XTrain']
    """
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):  # Skip HDF5 metadata
                continue
            try:
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    # Handle different data types
                    if item.dtype.char == 'U':  # Unicode strings
                        data[key] = [''.join(chr(c[0]) for c in f[item[0][0]][:].T)]
                    elif len(item.shape) == 2 and item.shape[0] == 1:
                        # Scalar or 1D array stored as row vector
                        data[key] = item[0, 0] if item.size == 1 else item[0, :]
                    else:
                        data[key] = item[:]
                        # Transpose if needed (MATLAB vs Python array ordering)
                        if len(data[key].shape) > 2:
                            data[key] = np.transpose(data[key])
            except Exception as e:
                print(f"Warning: Could not load {key}: {e}")
                continue
    return data

