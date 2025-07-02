import os, pickle, glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import zarr
from diffusion_mk2.model.normalization import DloDataProcessor, ActionDataProcessor




class DloDataset(Dataset):
    def __init__(self, dataset_path, num_points=15):
        self.num_points = num_points

        dataset_root = zarr.open(dataset_path, 'r')
        initial_shapes = dataset_root['data']['initial_shape'][:].reshape(-1, num_points, 3)
        final_shapes = dataset_root['data']['final_shape'][:].reshape(-1, num_points, 3)
        actions = dataset_root['data']['action'][:]

        assert len(initial_shapes) == len(final_shapes) == len(actions)

        self.data_samples, self.init_shape_nans, self.final_shape_nans = self.preprocess(initial_shapes, final_shapes, actions)


    def preprocess(self, initial_shapes, final_shapes, actions):
        initial_shapes_processor = DloDataProcessor(initial_shapes[:, :, :2])
        final_shapes_processor = DloDataProcessor(final_shapes[:, :, :2])
        actions_processor = ActionDataProcessor(actions, self.num_points)

        norm_factors = initial_shapes_processor.compute_normalize_factors_arrays()

        initial_shapes_processor.set_normalize_factors_arrays(*norm_factors)
        final_shapes_processor.set_normalize_factors_arrays(*norm_factors)
        actions_processor.set_normalize_factors_arrays(*norm_factors)

        initial_shapes_n, init_shapes_nans = initial_shapes_processor.preprocess()
        final_shapes_n, final_shapes_nans = final_shapes_processor.preprocess()
        actions_n = actions_processor.preprocess()

        samples = []
        for initial_shape_n, final_shape_n, action_n in zip(initial_shapes_n, final_shapes_n, actions_n):
            samples.append([
                torch.from_numpy(initial_shape_n.copy()).float(),
                torch.from_numpy(final_shape_n.copy()).float(),
                torch.from_numpy(action_n.copy()).float()
            ])
        return samples, init_shapes_nans, final_shapes_nans


    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]


if __name__ == "__main__":
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/shape_prediction_better.zarr.zip"
    dataset = DloDataset(dataset_path, num_points=15)
    print("Dataset length:", len(dataset), "init nans:", dataset.init_shape_nans, "final nans:", dataset.final_shape_nans)
