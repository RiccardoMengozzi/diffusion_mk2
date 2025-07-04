import numpy as np

class ActionDataProcessor():
    def __init__(self, action_data, num_points, is_first_idx, is_last_gripper):
        self.action_data = action_data
        self.num_points = num_points
        self.cs0_list = None
        self.csR_list = None
        self.is_first_idx = is_first_idx  # Whether the first component is an index
        self.is_last_gripper = is_last_gripper  # Whether the last component is the
        print("self.is_first_idx:", self.is_first_idx, "self.is_last_gripper:", self.is_last_gripper)

    def set_normalize_factors_arrays(self, cs0_list, csR_list):
        self.cs0_list = cs0_list
        self.csR_list = csR_list

    def normalize(self, action, csR, ):
        # Extract components - works for both 2D and 3D
        pos_start_idx, pos_last_idx = 0, -1
        if self.is_first_idx:
            idx = action[0]
            # Normalize index
            idx_n = idx / (self.num_points - 1.0)
            pos_start_idx = 1
        if self.is_last_gripper:
            gripper_state = action[-1]  # Last component is gripper state
            pos_last_idx = -2 

        pos = action[pos_start_idx:pos_last_idx]  # All spatial components (dx, dy) or (dx, dy, dz)
            
        dtheta = action[pos_last_idx+1] 
        print(f"pos_first_idx: {pos_start_idx}, pos_last_idx: {pos_last_idx}, pos_shape: {pos.shape}, self.is_first_idx: {self.is_first_idx}, self.is_last_gripper: {self.is_last_gripper}")

        
        # Normalize spatial components using rotation matrix
        pos_n = csR @ pos
        
        # Rotation remains unchanged
        theta_n = dtheta
        
        # Reconstruct normalized action
        if self.is_first_idx and self.is_last_gripper:
            return np.concatenate([[idx_n], pos_n, [theta_n], [gripper_state]])
        elif self.is_first_idx:
            return np.concatenate([[idx_n], pos_n, [theta_n]])
        elif self.is_last_gripper:
            return np.concatenate([pos_n, [theta_n], [gripper_state]])
        else:
            return np.concatenate([pos_n, [theta_n]])

        
    def denormalize(self, action_n, csR):
        pos_start_idx, pos_last_idx = 0, -1
        if self.is_first_idx:
            idx_n = action_n[0]
            # denormalize index
            idx = idx_n * (self.num_points - 1.0)
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
            return np.concatenate([[idx], pos, [theta], [gripper_state]])
        elif self.is_first_idx:
            return np.concatenate([[idx], pos, [theta]])
        elif self.is_last_gripper:
            return np.concatenate([pos, [theta], [gripper_state]])
        else:
            return np.concatenate([pos, [theta]])


    def preprocess(self):
        if self.cs0_list is None or self.csR_list is None:
            raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")
        
        processed_action_data = []
        for action, cs0, csR in zip(self.action_data, self.cs0_list, self.csR_list):
            action_n = self.normalize(action, csR)  # Note: removed cs0 parameter as it wasn't used
            processed_action_data.append(action_n)
        
        return np.array(processed_action_data)

class EEStateDataProcessor():
    def __init__(self, ee_state_data):
        self.ee_state_data = ee_state_data
        self.cs0_list = None
        self.csR_list = None

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

        
    def denormalize(self, ee_state_n, cs0, csR):
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

    def preprocess(self):
        if self.cs0_list is None or self.csR_list is None:
            raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")
        
        processed_ee_state_data = []
        for ee_state, cs0, csR in zip(self.ee_state_data, self.cs0_list, self.csR_list):
            ee_state_n = self.normalize(ee_state, cs0, csR)  # Note: removed cs0 parameter as it wasn't used
            processed_ee_state_data.append(ee_state_n)
        
        return np.array(processed_ee_state_data)
    

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