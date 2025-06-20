#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTStateDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data (obs, action) from a zarr storage
#@markdown - Normalizes each dimension of obs and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `obs`: shape (obs_horizon, obs_dim)
#@markdown  - key `action`: shape (pred_horizon, action_dim)

import numpy as np
import zarr
import torch

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        # [start_idx, end_idx) defines the entire episode
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        # [min_start, max_start] defines the observation-action sequence

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
        # [buffer_start_idx, buffer_end_idx] defines the data segment in the input data array
        # [sample_start_idx, sample_end_idx] defines the mapping to the output data that will be used in training
        # In the first segment, for example, if i have the "obs_horizon" of 2, i wont be able to have the previous observation, 
        # because that would be idx = -1, so that will be padded. in this case i will have:
        # buffer_start_idx = 0, buffer_end_idx = 16, sample_start_idx = 1, sample_end_idx = 16
        # Same thing at the end of the episode, if i have the "action_horizon" of 16, i wont be able to have the next 15 actions,
        # so that will be padded. in this case i will have:
        # buffer_start_idx = N - 15, buffer_end_idx = N, sample_start_idx = 0, sample_end_idx = 1, the next 15 actions will be padded 
        # with the last action (this if the episode finish exactly at the end of the dataset... if the episode finishes before, probably i
        # will need less padding, for example: 
        #  [25636 25650     0    14]
        #  [25637 25650     0    13]
        #  [25638 25650     0    12]
        #  [25639 25650     0    11]
        #  [25640 25650     0    10]
        #  [25641 25650     0     9]] 
        

    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]
        self.episode_ends = episode_ends
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample
    

if __name__ == "__main__":
    # Example usage
    dataset = PushTStateDataset(
        dataset_path="/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/teleop_pushing_dataset.zarr.zip",
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8
    )
    
    print("Dataset length:", len(dataset))
    sample = dataset[3]
    print("episode ends:", dataset.episode_ends)
    # print("Sample keys:", sample.keys())
    # print("Sample obs shape:", sample['obs'].shape)
    # print("Sample action shape:", sample['action'].shape)
    # print("Stats:", dataset.stats)
    # print("First 20 obs:", sample['obs'])
    # print("First 20 actions:", sample['action'][:20])