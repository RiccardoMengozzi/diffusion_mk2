import numpy as np


class ActionDataProcessor():
    def __init__(self, action_data, num_points):
        self.action_data = action_data
        self.num_points = num_points
        self.cs0_list = None
        self.csR_list = None

    def set_normalize_factors_arrays(self, cs0_list, csR_list):
        self.cs0_list = cs0_list
        self.csR_list = csR_list

    def normalize(self, action, csR):
        # Extract components - works for both 2D and 3D
        idx = action[0]
        spatial_components = action[1:-1]  # All spatial components (dx, dy) or (dx, dy, dz)
        dtheta = action[-1]  # Last component is always rotation
        
        # Normalize index
        idx_n = idx / (self.num_points - 1.0)
        
        # Normalize spatial components using rotation matrix
        spatial_components_n = csR @ spatial_components
        
        # Rotation remains unchanged
        theta_n = dtheta
        
        # Reconstruct normalized action
        return np.concatenate([[idx_n], spatial_components_n, [theta_n]])

        
    def denormalize(self, action_n, csR):
        # Extract components
        idx_n = action_n[0]
        spatial_components_n = action_n[1:-1]  # All spatial components
        dtheta_n = action_n[-1]
        
        # Denormalize index
        idx = idx_n * (self.num_points - 1.0)
        
        # Denormalize spatial components
        spatial_components = csR.T @ spatial_components_n
        
        # Rotation remains unchanged
        theta = dtheta_n
        
        # Reconstruct denormalized action
        return np.concatenate([[idx], spatial_components, [theta]])

    def preprocess(self):
        if self.cs0_list is None or self.csR_list is None:
            raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")
        
        processed_action_data = []
        for action, cs0, csR in zip(self.action_data, self.cs0_list, self.csR_list):
            action_n = self.normalize(action, csR)  # Note: removed cs0 parameter as it wasn't used
            processed_action_data.append(action_n)
        
        return np.array(processed_action_data)


class DloDataProcessor():
    def __init__(self, dlo_data):
        self.dlo_data = dlo_data 
        self.cs0_list = None
        self.csR_list = None

    def _compute_normalize_factors(self, dlo):
        # Works automatically for any number of spatial dimensions
        cs0 = np.mean(dlo, axis=0, keepdims=True)
        dlo_centered = dlo - cs0
        cov = dlo_centered.T @ dlo_centered
        eigval, eigvec = np.linalg.eig(cov)
        # Sort eigenvalues in descending order and arrange eigenvectors accordingly
        sorted_indices = np.argsort(eigval)[::-1]
        csR = eigvec[:, sorted_indices].T
        return cs0, csR
    
    def compute_normalize_factors_arrays(self):
        cs0_list = []
        csR_list = []
        for dlo in self.dlo_data:
            cs0, csR = self._compute_normalize_factors(dlo)
            cs0_list.append(cs0)
            csR_list.append(csR)
        return np.array(cs0_list), np.array(csR_list)

    def set_normalize_factors_arrays(self, cs0_list, csR_list):
        self.cs0_list = cs0_list
        self.csR_list = csR_list

    def get_normalize_factors_arrays(self):
        if self.cs0_list is not None and self.csR_list is not None:
            return self.cs0_list, self.csR_list
        else:
            raise ValueError("Normalization factors not set. Call compute_normalize_factors_arrays() first.")

    def _is_nan(self, dlo):
        return np.isnan(dlo).any() or np.isinf(dlo).any()
    
    def normalize(self, dlo, cs0, csR):
        # Matrix operations work for any number of dimensions
        dlo_n = (csR @ (dlo - cs0).T).T
        return dlo_n
            
    def denormalize(self, dlo, cs0, csR):
        # Matrix operations work for any number of dimensions
        dlo_dn = (csR.T @ dlo.T).T + cs0
        return dlo_dn

    def preprocess(self):
        if self.cs0_list is None or self.csR_list is None:
            raise ValueError("Normalization factors not set. Call compute_normalize_factors_arrays() first.")

        processed_dlo_data = []
        nans_counter = 0
        for dlo, cs0, csR in zip(self.dlo_data, self.cs0_list, self.csR_list):
            dlo_n = self.normalize(dlo, cs0, csR)
            if self._is_nan(dlo_n):
                nans_counter += 1
                continue
            processed_dlo_data.append(dlo_n)
        
        return np.array(processed_dlo_data), nans_counter