import os, pickle, glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import zarr

class DloAction:
    def __init__(self, num_points=16, scale_action=False):
        self.num_points = num_points
        self.scale_action = scale_action
        self.rot_check_flag = False
        self.cs0 = None
        self.csR = None
        self.angle_scale = np.pi / 4.0
        self.disp_scale = 0.1

    def set_normalize_factor(self, csR, cs0):
        self.csR = csR
        self.cs0 = cs0

    def set_rot_flag(self, rot_flag):
        self.rot_check_flag = rot_flag

    def normalize(self, dlo_0, a):

        a1 = self.normalize_action_idx(a[0])
        points_grasp, points_place = self.compute_edges_points_from_action(dlo_0, a)
        points_grasp_up = np.dot(self.csR, (points_grasp - self.cs0).T).T
        points_place_up = np.dot(self.csR, (points_place - self.cs0).T).T
        action_new = self.compute_action_from_edges_points(points_grasp_up, points_place_up)
        a23 = self.scale_disp(action_new[:2])
        a4 = self.scale_angle(action_new[-1])
        return np.stack([a1, a23[0], a23[1], a4], axis=-1)

    def normalize_action_idx(self, idx):
        if self.rot_check_flag:
            idx = (self.num_points - 2) - idx
        return idx / (self.num_points - 1.0)

    def scale_disp(self, disp):
        return disp / self.disp_scale if self.scale_action else disp

    def scale_angle(self, theta):
        return theta / self.angle_scale if self.scale_action else theta

    def denormalize(self, dlo_0_n, a):
        a1 = self.denormalize_action_idx(a[0])
        a23 = a[1:3] * self.disp_scale if self.scale_action else a[1:3]
        a4 = a[3] * self.angle_scale if self.scale_action else a[3]
        a_new = np.array([a1, *a23, a4])
        points_grasp, points_place = self.compute_edges_points_from_action(dlo_0_n, a_new)
        points_grasp_up = np.dot(self.csR.T, points_grasp.T).T + self.cs0
        points_place_up = np.dot(self.csR.T, points_place.T).T + self.cs0
        action_new = self.compute_action_from_edges_points(points_grasp_up, points_place_up)
        return np.stack([a1, action_new[0], action_new[1], action_new[2]], axis=-1)

    def denormalize_action_idx(self, idx):
        idx = idx * (self.num_points - 1.0)
        return (self.num_points - 2) - idx if self.rot_check_flag else idx

    def compute_edges_points_from_action(self, init_pos, action):
        idx = int(action[0])
        dtheta = action[3]
        node_0_pos = init_pos[idx, :]
        node_1_pos = init_pos[idx + 1, :]
        edge_pos = (node_1_pos + node_0_pos) / 2
        edge_dir = node_1_pos - node_0_pos
        edge_len = np.linalg.norm(edge_dir)
        edge_dir = edge_dir / edge_len
        new_edge_pos = edge_pos + np.array([action[1], action[2]])
        new_edge_dir = np.array([
            edge_dir[0] * np.cos(dtheta) - edge_dir[1] * np.sin(dtheta),
            edge_dir[0] * np.sin(dtheta) + edge_dir[1] * np.cos(dtheta)
        ])
        pos0_tgt = new_edge_pos - new_edge_dir * edge_len / 2
        pos1_tgt = new_edge_pos + new_edge_dir * edge_len / 2
        return np.array([node_0_pos, node_1_pos]), np.array([pos0_tgt, pos1_tgt])

    def compute_action_from_edges_points(self, points_grasp, points_place):
        center_grasp = points_grasp.mean(axis=0)
        center_place = points_place.mean(axis=0)
        dir_grasp = points_grasp[1] - points_grasp[0]
        dir_grasp /= np.linalg.norm(dir_grasp)
        dir_place = points_place[1] - points_place[0]
        dir_place /= np.linalg.norm(dir_place)
        angle = np.arctan2(dir_place[1], dir_place[0]) - np.arctan2(dir_grasp[1], dir_grasp[0])
        disp = center_place - center_grasp
        return np.array([disp[0], disp[1], angle])


class DloSample:
    def __init__(self, num_points=16, scale_action=True):
        self.dlo_action = DloAction(num_points, scale_action)
        self.num_points = num_points
        self.cs0 = self.csR = None
        self.rot_check_flag = None


    def normalize(self, dlo_0, dlo_1, action):
        self.cs0, self.csR = self.compute_normalize_factor(dlo_0)
        self.dlo_action.set_normalize_factor(self.csR, self.cs0)
        dlo_0_n = self.normalize_dlo(dlo_0)
        dlo_1_n = self.normalize_dlo(dlo_1)
        dlo_0_n, dlo_1_n = self.check_rot_and_flip(dlo_0_n, dlo_1_n)
        self.dlo_action.set_rot_flag(self.rot_check_flag)
        action_n = self.dlo_action.normalize(dlo_0, action)
        return dlo_0_n, dlo_1_n, action_n

    def compute_normalize_factor(self, dlo):
        cs0 = np.mean(dlo, axis=0, keepdims=True)
        dlo_centered = dlo - cs0
        cov = dlo_centered.T @ dlo_centered
        eigval, eigvec = np.linalg.eig(cov)
        csR = eigvec[:, np.argsort(eigval)[::-1]].T
        return cs0, csR

    def normalize_dlo(self, dlo):
        return (self.csR @ (dlo - self.cs0).T).T

    def check_rot_and_flip(self, dlo_0_n, dlo_1_n):
        self.rot_check_flag = dlo_0_n[0, 0] > 0.0
        return (dlo_0_n[::-1], dlo_1_n[::-1]) if self.rot_check_flag else (dlo_0_n, dlo_1_n)

    def denormalize(self, dlo_0_n, dlo_1_n, action_n):
        dlo_0 = self.denormalize_dlo(dlo_0_n)
        dlo_1 = self.denormalize_dlo(dlo_1_n)
        action = self.dlo_action.denormalize(dlo_0_n, action_n)
        return dlo_0, dlo_1, action

    def denormalize_dlo(self, dlo):
        if self.rot_check_flag:
            dlo = dlo[::-1]
        return (self.csR.T @ dlo.T).T + self.cs0


class DloDataset(Dataset, DloSample):
    def __init__(self, dataset_path, num_points=16, augment=False, scale_action=True):
        DloSample.__init__(self, num_points=num_points, scale_action=scale_action)
        self.augment = augment
        self.scale_action = scale_action
        self.num_points = num_points

        dataset_root = zarr.open(dataset_path, 'r')
        initial_shapes = dataset_root['data']['initial_shape'][:].reshape(-1, num_points, 3)
        final_shapes = dataset_root['data']['final_shape'][:].reshape(-1, num_points, 3)
        actions = dataset_root['data']['action'][:]
        print(actions)

        assert len(initial_shapes) == len(final_shapes) == len(actions)

        self.data_samples = self.preprocess(initial_shapes, final_shapes, actions)

    def preprocess(self, initial_shapes, final_shapes, actions):
        samples = []
        for dlo_0, dlo_1, action in zip(initial_shapes, final_shapes, actions):
            print(f"action1: {action}")
            dlo_0, dlo_1 = dlo_0[:, :2], dlo_1[:, :2]
            idx, dx, dy, dtheta = action
            if np.linalg.norm(dlo_0[0] - dlo_1[0]) > np.linalg.norm(dlo_0[0] - dlo_1[-1]):
                dlo_1 = np.flip(dlo_1, axis=0)
            dlo_0_n, dlo_1_n, action_n = self.normalize(dlo_0, dlo_1, [idx, dx, dy, dtheta])
            print(f"action2: {action_n}")
            samples.append([
                torch.from_numpy(dlo_0_n).float(),
                torch.from_numpy(dlo_1_n).float(),
                torch.from_numpy(action_n).float()
            ])
        return samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]


if __name__ == "__main__":
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/shape_prediction.zarr.zip"
    dataset = DloDataset(dataset_path, num_points=15)
    print("Dataset length:", len(dataset))
