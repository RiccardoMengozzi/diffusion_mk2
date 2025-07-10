import numpy as np

def get_data_stats(data):
    """Compute min and max for each dimension of the data.
    If data has shape (..., 3), flatten all but last dim to get global min/max per x,y,z."""
    if data.ndim >= 2 and data.shape[-1] == 3:
        # appiattisci tutte le dimensioni iniziali in un unico asse
        flat = data.reshape(-1, 3)  
        min_vals = np.min(flat, axis=0)  # un solo min per x,y,z
        max_vals = np.max(flat, axis=0)  # un solo max per x,y,z
    else:
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
    return {"min": min_vals, "max": max_vals}

def normalize_data(data, stats, keep_zero_centered):
    """Normalize data to using the provided stats."""
    if keep_zero_centered:
        range = np.max([np.abs(stats["min"]), np.abs(stats["max"])], axis=0)
        ndata = data / range
        return ndata
    # normalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def denormalize_data(ndata, stats, keep_zero_centered):
    """Unnormalize data to original scale using the provided stats."""
    if keep_zero_centered:
        range = np.max([np.abs(stats["min"]), np.abs(stats["max"])], axis=0)
        data = ndata * range
        return data
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data