import numpy as np
import zarr
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CREATE = False  # Set to True to create a new dataset, False to read an existing one
project_dir = os.path.dirname(os.path.abspath(__file__))

class ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        self.store  = zarr.open(path, mode="r")
        self.states = self.store["states"]
        self.actions = self.store["actions"]
        assert len(self.states) == len(self.actions), "Mismatched lengths"
        self.transforms = transforms

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]     # OBSERVATION: numpy array (O+1, 2): O+1: Objects state dimension + robot ee state, 2: x,y position 
        action = self.actions[idx]  # ACTION: numpy array (2): x, y desired ee position

        state = torch.from_numpy(state).float()
        action = torch.tensor(action).float()

        if self.transforms:
            state = self.transforms(state)

        return state, action

if __name__ == "__main__":
    if CREATE:
        # 1.1 Define output directory
        store_path = f"{project_dir}/my_data.zarr"
        if os.path.exists(store_path):
            # remove old store if rerunning
            import shutil; shutil.rmtree(store_path)

        # 1.2 Create a Zarr group (directory)
        root = zarr.open(store_path, mode="w")

        num_samples =   1_000
        object_dim = 10
        state_shape = (object_dim + 1, 2)
        action_shape = (2,)

        # Chunking: e.g. 100 samples per chunk
        state_chunks = (100, *state_shape)
        action_chunks = (100, *action_shape)

        root.create_dataset(
            name="states",
            shape=(num_samples, *state_shape),
            chunks=state_chunks,
            dtype="uint8",
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )

        root.create_dataset(
            name="actions",
            shape=(num_samples, *action_shape),
            chunks=action_chunks,
            dtype="int64",
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )

        # 1.4 Fill with dummy data (or your real arrays)
        #     Here we write random, but you’d write your prepared numpy arrays.
        root["states"][:] = np.random.rand(num_samples, *(state_shape)).astype("float32")
        root["actions"][:] = np.random.rand(num_samples, *(action_shape)).astype("float32")

        print("Zarr store created at", store_path)
    else:
        # ------------------------------------------------------------
        # Reading branch: load states + actions in batches and inspect
        # ------------------------------------------------------------
        batch_size = 8

        dataset = ZarrDataset(os.path.join(project_dir, "my_data.zarr"))
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Get one batch
        states_batch, actions_batch = next(iter(loader))
        # states_batch: (batch_size, O+1, 2)
        # actions_batch: (batch_size, 2)

        # Simple sanity check: print shapes and a small sample
        print(f"states_batch.shape = {states_batch.shape}")   # expect (8, O+1, 2)
        print(f"actions_batch.shape = {actions_batch.shape}") # expect (8, 2) 
        print("\nExample [first sample] in batch:")
        print(" state  =", states_batch[0])
        print(" action =", actions_batch[0])