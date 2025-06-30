import os, pickle, glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import zarr


class DloDataset(Dataset):
    def __init__(self, dataset_path, num_points=16, augment=False, scale_action=True):
        self.num_points = num_points
        self.augment = augment
        self.scale_action = scale_action

        # Load Zarr data
        dataset_root = zarr.open(dataset_path, 'r')
        initial_shapes = dataset_root['data']['initial_shape'][:]
        final_shapes = dataset_root['data']['final_shape'][:]
        actions = dataset_root['data']['action'][:]


        assert len(initial_shapes) == len(final_shapes) == len(actions)

        initial_shapes = initial_shapes.reshape(-1, num_points, 3).astype(np.float32)
        final_shapes = final_shapes.reshape(-1, num_points, 3).astype(np.float32)

        self.data_samples = self.preprocess(initial_shapes, final_shapes, actions)

    def preprocess(self, initial_shapes, final_shapes, actions):
        data_samples = []
        for dlo_0, dlo_1, action in zip(initial_shapes, final_shapes, actions):
            dlo_0 = dlo_0[:, :2]
            dlo_1 = dlo_1[:, :2]
            idx, dx, dy, dtheta = action

            # Flip final shape if necessary
            if np.linalg.norm(dlo_0[0] - dlo_1[0]) > np.linalg.norm(dlo_0[0] - dlo_1[-1]):
                dlo_1 = np.flip(dlo_1, axis=0)

            dlo_0_n, dlo_1_n, action_n = self.normalize(dlo_0, dlo_1, [idx, dx, dy, dtheta])

            data_samples.append([
                torch.from_numpy(dlo_0_n).float(),
                torch.from_numpy(dlo_1_n).float(),
                torch.from_numpy(action_n).float()
            ])

        return data_samples

    def compute_normalize_factor(self, dlo):
        cs0 = np.mean(dlo, axis=0, keepdims=True)
        dlo_centered = dlo - cs0
        cov = dlo_centered.T @ dlo_centered
        eigval, eigvec = np.linalg.eig(cov)
        csR = eigvec[:, np.argsort(eigval)[::-1]].T
        return cs0, csR

    def normalize_dlo(self, dlo, cs0, csR):
        return (csR @ (dlo - cs0).T).T

    def normalize(self, dlo_0, dlo_1, action):
        cs0, csR = self.compute_normalize_factor(dlo_0)

        dlo_0_n = self.normalize_dlo(dlo_0, cs0, csR)
        dlo_1_n = self.normalize_dlo(dlo_1, cs0, csR)

        # Check rotation flag
        rot_check_flag = dlo_0_n[0, 0] > 0.0
        if rot_check_flag:
            dlo_0_n = dlo_0_n[::-1]
            dlo_1_n = dlo_1_n[::-1]
            action[0] = (self.num_points - 2) - action[0]

        idx_n = action[0] / (self.num_points - 1.0)
        dx, dy = action[1], action[2]
        theta = action[3]

        if self.scale_action:
            dx /= 0.1
            dy /= 0.1
            theta /= (np.pi / 4.0)

        return dlo_0_n, dlo_1_n, np.array([idx_n, dx, dy, theta])

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]


if __name__ == "__main__":
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/shape_prediction.zarr.zip"
    dataset = DloDataset(dataset_path, num_points=15)
    print("Dataset length:", len(dataset))
