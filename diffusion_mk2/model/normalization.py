import numpy as np


class ActionDataProcessor():
    def __init__(self, action_data, num_points, is_first_idx, is_last_gripper):
        self.action_data = action_data
        self.num_points = num_points
        self.cs0_list = None
        self.csR_list = None
        self.scale_min = None
        self.scale_max = None
        self.is_first_idx = is_first_idx  # Whether the first component is an index
        self.is_last_gripper = is_last_gripper  # Whether the last component is the

    def set_normalize_factors_arrays(self, cs0_list, csR_list):
        self.cs0_list = cs0_list
        self.csR_list = csR_list

    def normalize(self, action, csR, ):
        # Extract components - works for both 2D and 3D
        pos_start_idx, pos_last_idx = 0, -1
        if self.is_first_idx:
            idx = action[0]  # First component is an index
            pos_start_idx = 1
        if self.is_last_gripper:
            gripper_state = action[-1]  # Last component is gripper state
            pos_last_idx = -2 

        pos = action[pos_start_idx:pos_last_idx]  # All spatial components (dx, dy) or (dx, dy, dz)
            
        dtheta = action[pos_last_idx+1] 

        
        # Normalize spatial components using rotation matrix
        pos_n = csR @ pos
        
        # Rotation remains unchanged
        theta_n = dtheta
        
        # Reconstruct normalized action
        if self.is_first_idx and self.is_last_gripper:
            return np.concatenate([[idx], pos_n, [theta_n], [gripper_state]])
        elif self.is_first_idx:
            return np.concatenate([[idx], pos_n, [theta_n]])
        elif self.is_last_gripper:
            return np.concatenate([pos_n, [theta_n], [gripper_state]])
        else:
            return np.concatenate([pos_n, [theta_n]])

        
    def denormalize(self, action_n, descale, csR=None, idx=None):
        """
        If you give cs0 and csR, they will be used for denormalization.
        otherwise u can give the idx, that will be used to get the cs0 and csR from the lists.
        """
        if csR is None:
            if self.csR_list is None:
                raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")
            if idx is None:
                raise ValueError("Either cs0 and csR must be provided or an index must be given.")
            csR = self.csR_list[idx]

        if descale:
            if self.scale_min is None or self.scale_max is None:
                raise ValueError("scaling range not set!!!")
            action_n = self.descale(action_n, self.scale_min, self.scale_max)

        pos_start_idx, pos_last_idx = 0, -1
        if self.is_first_idx:
            idx_n = action_n[0]
            # denormalize index
            pos_start_idx = 1
        if self.is_last_gripper:
            gripper_state = action_n[-1]  # Last component is gripper state
            pos_last_idx = -2 

        pos_n = action_n[pos_start_idx:pos_last_idx]  # All spatial components (dx, dy) or (dx, dy, dz)
            
        dtheta_n = action_n[pos_last_idx+1] 
        
        # Denormalize spatial components
        pos = csR.T @ pos_n
        
        # Rotation remains unchanged
        theta = dtheta_n
        
        # Reconstruct denormalized action
        if self.is_first_idx and self.is_last_gripper:
            return np.concatenate([[idx_n], pos, [theta], [gripper_state]])
        elif self.is_first_idx:
            return np.concatenate([[idx_n], pos, [theta]])
        elif self.is_last_gripper:
            return np.concatenate([pos, [theta], [gripper_state]])
        else:
            return np.concatenate([pos, [theta]])


    def scale(self, action, min, max):

        # Scale to [0, 1]: (data - min) / (max - min)
        scaled_action = (action - min) / (max - min)
        
        # Convert to [-1, 1]: [0, 1] -> [-1, 1]
        scaled_action = scaled_action * 2 - 1

        return scaled_action

    def descale(self, action_scaled, min=None, max=None):
        if min is None or max is None:
            if self.scale_min is None or self.scale_max is None:
                raise ValueError("scaling range not set!!!")
            min, max = self.scale_min, self.scale_max
        # Extract components
        # Convert from [-1, 1] to [0, 1]
        action = (action_scaled + 1) / 2
        
        # Scale back to original range: [0, 1] -> [min, max]
        action = action * (max - min) + min
        return action

    def preprocess(self):
        if self.cs0_list is None or self.csR_list is None:
            raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")

        processed_action_data = []
        for action, cs0, csR in zip(self.action_data, self.cs0_list, self.csR_list):
            action_n = self.normalize(action, csR)  # Note: removed cs0 parameter as it wasn't used
            processed_action_data.append(action_n)

        processed_action_data = np.array(processed_action_data)
        self.scale_min = np.min(processed_action_data, axis=0)
        self.scale_max = np.max(processed_action_data, axis=0)


        scaled_processed_action_data = []
        for action in processed_action_data:
            action = self.scale(action, self.scale_min, self.scale_max)
            scaled_processed_action_data.append(action)

        return np.array(scaled_processed_action_data)

class EEStateDataProcessor():
    def __init__(self, ee_state_data):
        self.ee_state_data = ee_state_data
        self.cs0_list = None
        self.csR_list = None
        self.scale_min = None
        self.scale_max = None

 

    def set_normalize_factors_arrays(self, cs0_list, csR_list):
        self.cs0_list = cs0_list
        self.csR_list = csR_list

    def normalize(self, ee_state, cs0, csR):
        # Extract components - works for both 2D and 3D
        pos = ee_state[0:-2]  # All spatial components (x, y) or (x, y, z) and theta
        theta = ee_state[-2]  # second last component is always rotation, last component is gripper state
        gripper_state = ee_state[-1]  # Last component is gripper state (open/close)

    
        pos_centered = pos - cs0     # subtract the mean/centroid
        pos_n = (csR @ pos_centered.T).T  
        pos_n = pos_n.flatten()  # Ensure it's a flat array

        # Rotation remains unchanged
        theta_n = theta
        # Reconstruct normalized action
        return np.concatenate([pos_n, [theta_n], [gripper_state]])  # Keep gripper state unchanged

        
    def denormalize(self, ee_state_n, descale, cs0=None, csR=None, idx=None):
        """
        If you give cs0 and csR, they will be used for denormalization.
        otherwise u can give the idx, that will be used to get the cs0 and csR from the lists.
        """
        if cs0 is None or csR is None:
            if self.cs0_list is None or self.csR_list is None:
                raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")
            if idx is None:
                raise ValueError("Either cs0 and csR must be provided or an index must be given.")
            cs0 = self.cs0_list[idx]
            csR = self.csR_list[idx]

        if descale:
            if self.scale_min is None or self.scale_max is None:
                raise ValueError("scaling range not set!!!")
            ee_state_n = self.descale(ee_state_n, self.scale_min, self.scale_max)

        # Extract components
        pos_n = ee_state_n[0:-2]  # All spatial components
        theta_n = ee_state_n[-2]
        gripper_state = ee_state_n[-1]  # Last component is gripper state (open/close)
        
        pos_centered = (csR.T @ pos_n.T).T  
        # inverse-translate
        pos = pos_centered + cs0
        pos = pos.flatten()  # Ensure it's a flat array

        # Rotation remains unchanged
        theta = theta_n
        
        # Reconstruct denormalized action
        return np.concatenate([pos, [theta], [gripper_state]])  # Keep gripper state unchanged
    
    def scale(self, ee_state, min, max):
        # Scale to [0, 1]: (data - min) / (max - min)
        scaled_ee_state = (ee_state - min) / (max - min)
        # Convert to [-1, 1]: [0, 1] -> [-1, 1]
        scaled_ee_state = scaled_ee_state * 2 - 1

        return scaled_ee_state

    def descale(self, ee_state_scaled, min=None, max=None):
        if min is None or max is None:
            if self.scale_min is None or self.scale_max is None:
                raise ValueError("scaling range not set!!!")
            min, max = self.scale_min, self.scale_max
        # Extract components
        # Convert from [-1, 1] to [0, 1]
        ee_state = (ee_state_scaled + 1) / 2
        
        # Scale back to original range: [0, 1] -> [min, max]
        ee_state = ee_state * (max - min) + min
        return ee_state

    def preprocess(self):
        if self.cs0_list is None or self.csR_list is None:
            raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")

        processed_ee_state_data = []
        for ee_state, cs0, csR in zip(self.ee_state_data, self.cs0_list, self.csR_list):
            ee_state_n = self.normalize(ee_state, cs0, csR)  # Note: removed cs0 parameter as it wasn't used
            processed_ee_state_data.append(ee_state_n)

        processed_ee_state_data = np.array(processed_ee_state_data)
        self.scale_min = np.min(processed_ee_state_data, axis=0)
        self.scale_max = np.max(processed_ee_state_data, axis=0)


        scaled_processed_ee_state_data = []
        for ee_state in processed_ee_state_data:
            ee_state = self.scale(ee_state, self.scale_min, self.scale_max)
            scaled_processed_ee_state_data.append(ee_state)

        return np.array(scaled_processed_ee_state_data)
    
    

class DloDataProcessor():
    def __init__(self, dlo_data):
        self.dlo_data = dlo_data 
        self.cs0_list = None
        self.csR_list = None

    def _compute_normalize_factors(self, dlo):
        cs0 = np.mean(dlo, axis=0, keepdims=True)
        dlo_centered = dlo - cs0
        
        # Check DLO linearity
        dlo_range = np.max(dlo, axis=0) - np.min(dlo, axis=0)
        is_linear = self._is_dlo_linear(dlo_centered, dlo_range)
        
        if is_linear:
            # For linear DLO, use the DLO's main axis as primary axis
            csR = self._compute_linear_coordinate_system(dlo_centered)
        else:
            # For non-linear DLO, use PCA
            csR = self._compute_pca_coordinate_system(dlo_centered)
        
        return cs0, csR

    def _is_dlo_linear(self, dlo_centered, dlo_range, linear_threshold=1000):
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
        
        return condition_number > linear_threshold  # Threshold for linearity, e.g., 1000

    def _compute_linear_coordinate_system(self, dlo_centered):
        """Compute coordinate system for linear DLO."""
        # Create coordinate system aligned with main axis
        csR = np.eye(3)
        
        # Compute actual direction of the DLO
        first_point = dlo_centered[0]
        last_point = dlo_centered[-1]
        direction = last_point - first_point
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        # Align first axis with DLO direction
        csR[0] = direction
        
        # Create orthogonal axes
        if abs(direction[2]) < 0.9:  # Not aligned with Z
            csR[1] = np.cross(direction, [0, 0, -1]) #-1 otherwise z becomes negative
        else:  # Aligned with Z, use X
            csR[1] = np.cross(direction, [1, 0, 0])
        
        csR[1] = csR[1] / (np.linalg.norm(csR[1]) + 1e-6)
        csR[2] = np.cross(csR[0], csR[1])
        
        return csR

    def _compute_pca_coordinate_system(self, dlo_centered):
        """Compute PCA-based coordinate system."""
        cov = dlo_centered.T @ dlo_centered
        eigval, eigvec = np.linalg.eig(cov)
        
        sorted_indices = np.argsort(eigval)[::-1]
        csR = eigvec[:, sorted_indices].T
        
        # Anti-flip adjustments
        if np.linalg.det(csR) < 0:
            csR[-1, :] *= -1
        
        for i in range(csR.shape[0]):
            j = np.argmax(np.abs(csR[i, :]))
            if csR[i, j] < 0:
                csR[i, :] *= -1
        
        return csR
    
    def compute_normalize_factors_arrays(self):
        cs0_list = []
        csR_list = []
        for dlo in self.dlo_data:
            cs0, csR = self._compute_normalize_factors(dlo)
            cs0_list.append(cs0)
            csR_list.append(csR)
        cs0_arr = np.vstack(cs0_list)
        csR_arr = np.stack(csR_list)


        # Step 2: enforce temporal sign consistency using frame 0 as reference
        R_ref = csR_arr[0].copy()
        for t in range(1, len(csR_arr)):
            R_cur = csR_arr[t]
            for k in range(R_cur.shape[0]):
                # if current axis is more opposite than aligned, flip sign
                if np.dot(R_ref[k], R_cur[k]) < 0:
                    R_cur[k] *= -1
            csR_arr[t] = R_cur

        self.cs0_list = cs0_arr
        self.csR_list = csR_arr
        return cs0_arr, csR_arr


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
            
    def denormalize(self, dlo, cs0=None, csR=None, idx=None):
        """
        If you give cs0 and csR, they will be used for denormalization.
        otherwise u can give the idx, that will be used to get the cs0 and csR from the lists.
        """
        if cs0 is None or csR is None:
            if self.cs0_list is None or self.csR_list is None:
                raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")
            if idx is None:
                raise ValueError("Either cs0 and csR must be provided or an index must be given.")
            cs0 = self.cs0_list[idx]
            csR = self.csR_list[idx]

        # Matrix operations work for any number of dimensions
        dlo = dlo.squeeze()  # Ensure dlo is 2D
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