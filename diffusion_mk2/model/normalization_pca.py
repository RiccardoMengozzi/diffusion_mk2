import numpy as np

def get_data_stats(data, dlo_dim=None):
    """Compute min and max for each dimension of the data."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return {"min": min_vals, "max": max_vals}



def compute_pca_coordinate_system(dlo_centered):
    """Compute PCA-based coordinate system ensuring right-handed system."""
    cov = dlo_centered.T @ dlo_centered
    eigval, eigvec = np.linalg.eig(cov)
    
    sorted_indices = np.argsort(eigval)[::-1]
    csR = eigvec[:, sorted_indices].T
    
    # CRITICAL: Ensure right-handed coordinate system
    if np.linalg.det(csR) < 0:
        # If determinant is negative, we have a left-handed system
        # Flip the last axis to make it right-handed
        csR[-1, :] *= -1
    
    # Optional: Additional consistency checks to reduce arbitrary flipping
    # Make sure the first principal component points in a consistent direction
    # For example, ensure it points towards the "positive" end of the shape
    
    # Method 1: Align with the vector from first to last point
    if dlo_centered.shape[0] > 1:
        main_direction = dlo_centered[-1] - dlo_centered[0]
        if np.dot(csR[0], main_direction) < 0:
            csR[0] *= -1
    
    # Method 2: Or align with centroid direction (for more stable results)
    # centroid_direction = np.mean(dlo_centered, axis=0)
    # if np.dot(csR[0], centroid_direction) < 0:
    #     csR[0] *= -1
    
    # Re-orthogonalize if needed (though usually not necessary)
    # csR = np.linalg.qr(csR.T)[0].T
    
    return csR

def compute_normalize_factors(dlo):
    """Compute normalization factors ensuring consistent orientation."""
    cs0 = np.mean(dlo, axis=0, keepdims=True)
    dlo_centered = dlo - cs0
    
    # Check DLO linearity
    dlo_range = np.max(dlo, axis=0) - np.min(dlo, axis=0)
    is_linear = is_dlo_linear(dlo_centered, dlo_range)
    
    if is_linear:
        # For linear DLO, use the DLO's main axis as primary axis
        csR = compute_linear_coordinate_system(dlo_centered)
    else:
        # For non-linear DLO, use PCA with consistent orientation
        csR = compute_pca_coordinate_system(dlo_centered)
    
    return cs0, csR

def compute_linear_coordinate_system(dlo_centered):
    """Compute coordinate system for linear DLO with consistent orientation."""
    
    # Check if the data is 2D or 3D
    is_2d = dlo_centered.shape[1] == 2
    
    # Create coordinate system aligned with the main axis
    csR = np.eye(dlo_centered.shape[1])
    
    # Compute direction of the DLO (from first to last point)
    first_point = dlo_centered[0]
    last_point = dlo_centered[-1]
    direction = last_point - first_point
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    
    if is_2d:
        # Align first axis with DLO direction
        csR[0] = direction
        
        # Compute the perpendicular vector (orthogonal to the direction)
        csR[1] = np.array([-direction[1], direction[0]])  # Perpendicular in 2D
        csR[1] = csR[1] / (np.linalg.norm(csR[1]) + 1e-6)
        
        # Ensure right-handed system (determinant should be positive)
        if np.linalg.det(csR) < 0:
            csR[1] *= -1
    else:
        # In 3D, align first axis with DLO direction
        csR[0] = direction
        
        # Create orthogonal axes
        if abs(direction[2]) < 0.9:  # Not aligned with Z axis
            csR[1] = np.cross(direction, [0, 0, 1])  # Cross with Z axis
        else:  # Aligned with Z, use X axis
            csR[1] = np.cross(direction, [1, 0, 0])
        
        csR[1] = csR[1] / (np.linalg.norm(csR[1]) + 1e-6)
        csR[2] = np.cross(csR[0], csR[1])  # Third axis
        
        # Ensure right-handed system
        if np.linalg.det(csR) < 0:
            csR[2] *= -1

    return csR

def is_dlo_linear(dlo_centered, dlo_range, linear_threshold=1000):
    """Check if DLO is approximately linear."""
    # Method 1: Check if one dimension dominates
    max_range = np.max(dlo_range)
    other_ranges = dlo_range[dlo_range != max_range]
    ratio = max_range / (np.mean(other_ranges) + 1e-6)
    
    if ratio > 10:  # One dimension is much larger
        return True
    
    # Method 2: Check PCA condition number
    cov = dlo_centered.T @ dlo_centered
    eigval = np.linalg.eigvals(cov)
    condition_number = eigval.max() / (eigval.min() + 1e-12)
    
    return condition_number > linear_threshold



def normalize_pca(data, cs0, csR, rotation_only=False):
    # Matrix operations work for any number of dimensions
    if rotation_only:
        data_n = (csR @ data.T).T
        return data_n
    
    data_n = (csR @ (data - cs0).T).T
    return data_n.squeeze()


def denormalize_pca(data_n, cs0, csR, rotation_only=False):
    # Matrix operations work for any number of dimensions
    if rotation_only:
        data = (csR.T @ data_n.T).T
        return data
    
    data = (csR.T @ data_n.T).T + cs0
    return data.squeeze()

def denormalize_pca_batch(data_n, cs0, csR, rotation_only=False):
    data = []
    for dn in data_n:
        if rotation_only:
            d = (csR.T @ dn.T).T
        else:
            d = (csR.T @ dn.T).T + cs0
        data.append(d)
    return np.array(data)


def normalize_min_max(data, min, max):
    data_n = (data - min) / (max - min)
    data_n = data_n * 2 - 1  # Normalize to [-1, 1]
    return data_n

def denormalize_min_max(data_n, min, max):
    data_n = (data_n + 1) / 2  # Normalize to [0, 1]
    data = data_n * (max - min) + min
    return data

def denormalize_min_max_batch(data_n, min, max):
    data = []
    for dn in data_n:
        d = (dn + 1) / 2 * (max - min) + min
        data.append(d)
    return np.array(data)