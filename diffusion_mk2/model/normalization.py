import numpy as np
import torch


class ActionDataProcessor():
    def __init__(self, action_data, num_points):
        self.action_data = action_data
        self.num_points = num_points
        self.cs0_list = None
        self.csR_list = None

    def set_normalize_factors_arrays(self, cs0_list, csR_list):
        self.cs0_list = cs0_list
        self.csR_list = csR_list

    def normalize(self, action, cs0, csR):
        idx, dx, dy, dtheta = action[0], action[1], action[2], action[3]
        idx_n = idx / (self.num_points - 1.0)
        disp_2d = np.array([dx, dy])
        disp_2d_n = csR @ disp_2d
        theta_n = dtheta

        return np.array([idx_n, disp_2d_n[0], disp_2d_n[1], theta_n])

        
    def denormalize(self, action_n, cs0, csR):
        idx_n, dx_n, dy_n, dtheta_n = action_n[0], action_n[1], action_n[2], action_n[3]
        idx = idx_n * (self.num_points - 1.0)
        disp_2d_n = np.array([dx_n, dy_n])
        disp_2d = csR.T @ disp_2d_n
        theta = dtheta_n

        return np.array([idx, disp_2d[0], disp_2d[1], theta])

    def preprocess(self):
        if self.cs0_list is None or self.csR_list is None:
            raise ValueError("Normalization factors not set. Call set_normalize_factors_arrays() first.")
        
        processed_action_data = []
        for action, cs0, csR in zip(self.action_data, self.cs0_list, self.csR_list):
            action_n = self.normalize(action, cs0, csR)
            processed_action_data.append(action_n)
        
        return np.array(processed_action_data)

class DloDataProcessor():
    def __init__(self, dlo_data):
        self.dlo_data = dlo_data 
        self.cs0_list = None
        self.csR_list = None

    def _compute_normalize_factors(self, dlo):
        cs0 = np.mean(dlo, axis=0, keepdims=True)
        dlo_centered = dlo - cs0
        cov = dlo_centered.T @ dlo_centered
        eigval, eigvec = np.linalg.eig(cov)
        csR = eigvec[:, np.argsort(eigval)[::-1]].T
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
        if self.cs0_list and self.csR_list:
            return self.cs0_list, self.csR_list
        else:
            raise ValueError("Normalization factors not set. Call compute_normalize_factors_arrays() first.")

    def _is_nan(self, dlo):
        return np.isnan(dlo).any() or np.isinf(dlo).any()
    

    def normalize(self, dlo, cs0, csR):
        dlo_n = (csR @ (dlo - cs0).T).T
        return dlo_n
            
    def denormalize(self, dlo, cs0, csR):
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
