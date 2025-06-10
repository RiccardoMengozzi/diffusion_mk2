import numpy as np
import zarr
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CREATE = False  # Set to True to create a new dataset, False to read an existing one
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        self.store  = zarr.open(path, mode="r")
        self.images = self.store["images"]
        self.labels = self.store["labels"]
        assert len(self.images) == len(self.labels), "Mismatched lengths"
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]       # numpy array (C,H,W)
        label = int(self.labels[idx])  # scalar

        img = torch.from_numpy(img).float().div(255.0)  # normalize to [0,1]
        label = torch.tensor(label, dtype=torch.long)

        if self.transforms:
            img = self.transforms(img)

        return img, label

if __name__ == "__main__":
    if CREATE:
        # 1.1 Define output directory
        store_path = os.path.join(WORKSPACE_DIR, "zarr_data/my_data.zarr")
        if os.path.exists(store_path):
            # remove old store if rerunning
            import shutil; shutil.rmtree(store_path)

        # 1.2 Create a Zarr group (directory)
        root = zarr.open(store_path, mode="w")


        # 1.3 Create one or more arrays inside the group
        #      Here: 10 000 samples of 3×64×64 images and a matching label vector.
        num_samples =   1_000
        img_shape   = (3, 64, 64)

        # Chunking: e.g. 100 samples per chunk
        chunks = (100, *img_shape)

        root.create_dataset(
            name="images",
            shape=(num_samples, *img_shape),
            chunks=chunks,
            dtype="uint8",
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )

        root.create_dataset(
            name="labels",
            shape=(num_samples,),
            chunks=(100,),
            dtype="int64",
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )

        # 1.4 Fill with dummy data (or your real arrays)
        #     Here we write random, but you’d write your prepared numpy arrays.
        root["images"][:] = np.random.randint(0, 256, size=(num_samples, *img_shape), dtype="uint8")
        root["labels"][:] = np.random.randint(0, 10, size=(num_samples,), dtype="int64")

        print("Zarr store created at", store_path)
    else:
        n = 9  
        grid_size = int(n**0.5)

        # Apri dataset e DataLoader con batch_size=n
        dataset = ZarrDataset(os.path.join(WORKSPACE_DIR, "zarr_data/my_data.zarr"))
        loader = DataLoader(dataset, batch_size=n, shuffle=True)

        # Prendi un batch di n immagini
        imgs_tensor, labels = next(iter(loader))  
        # imgs_tensor: (n, C, H, W)

        # Converti in (n, H, W, C) numpy array
        imgs = imgs_tensor.permute(0, 2, 3, 1).numpy()

        # Crea la figura a griglia
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
        for idx, ax in enumerate(axes.flatten()):
            ax.imshow(imgs[idx])
            ax.set_title(f"Label: {labels[idx].item()}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()