import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusion_mk2.model.normalization import DloDataProcessor, ActionDataProcessor
from diffusion_mk2.dataset.pusht_state_dataset import PushTStateDataset

# Configure numpy printing
np.set_printoptions(precision=8, suppress=True, linewidth=100, threshold=1000)

def plot_shape(ax, dlo, action, title=None):
    """Plot 2D shape data."""

    init_idx_pos = dlo[int(action[0])]

    ax.plot(dlo[:, 0], dlo[:, 1], "o-", label="dlo")
    print("action:", action)
    ax.arrow(init_idx_pos[0], 
             init_idx_pos[1], 
             action[1] - init_idx_pos[0], 
             action[2] - init_idx_pos[1], 
             head_width=0.005, 
             head_length=0.005, 
             fc='red', 
             ec='red', 
             label='action',
             length_includes_head=True)
    
    if title:
        ax.set_title(title)
    ax.legend()
    ax.axis("equal")





def load_dataset(dataset_path, obs_ee_dim, obs_shape_dim, obs_target_dim):
    """Load and return dataset components."""
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=1,
        obs_horizon=1,
        action_horizon=1,
        obs_ee_dim=obs_ee_dim,
        obs_dlo_dim=obs_shape_dim,
        obs_target_dim=obs_target_dim,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataset, dataloader


def extract_shapes(dataset_path, dataloader, obs_ee_dim, obs_shape_dim):
    """Extract original and processed shape data."""
    # Load original shapes from zarr
    dataset_root = zarr.open(dataset_path, "r")
    original_shapes = dataset_root["data"]["state"][
        :, obs_ee_dim : obs_ee_dim + obs_shape_dim
    ]
    original_idxs = dataset_root["data"]["idx"]
    original_actions = dataset_root["data"]["action"][:, :-1]  # Exclude the last element (gripper)

    print("original idxs shape:", original_idxs.shape)
    print("original actions shape:", original_actions.shape)
    original_actions = np.concatenate([np.expand_dims(original_idxs, axis=1), original_actions], axis=1)


    print("original_actions:", original_actions.shape)
    original_shapes = original_shapes.reshape(-1, 15, 3)
    # Extract shapes from dataloader
    processed_shapes = []
    processed_actions = []
    for batch in dataloader:
        processed_shapes.append(batch["obs"])
        processed_actions.append(batch["action"])

    processed_shapes = np.array(processed_shapes).squeeze()
    processed_actions = np.array(processed_actions).squeeze()
    processed_shapes = processed_shapes[:, obs_ee_dim : obs_ee_dim + obs_shape_dim]

    return (
        original_shapes.astype(np.float32),
        original_actions,
        processed_shapes,
        processed_actions,
    )


def visualize_normalization(dlo_processor, 
                            action_processor,
                            shape, 
                            action,
                            index):


    """Visualize normalization process for a single shape."""
    cs0, csR = dlo_processor._compute_normalize_factors(shape)
    shape_n = dlo_processor.normalize(shape, cs0, csR)
    shape_dn = dlo_processor.denormalize(shape_n, cs0, csR)
    action_n = action_processor.normalize(action, csR)
    action_dn = action_processor.denormalize(action_n, csR)

    print("original action:", action)
    print("normalized action:", action_n)
    print("denormalized action:", action_dn)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_shape(axes[0], shape, action, "Original")
    plot_shape(axes[1], shape_n, action_n, "Normalized")
    plot_shape(axes[2], shape_dn, action_dn, "Denormalized")

    plt.suptitle(f"Shape {index} - Normalization Process")
    plt.tight_layout()
    plt.show()


def main():
    # Configuration
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/pushing_dataset_test_first.zarr.zip"
    obs_ee_dim = 5
    obs_shape_dim = 45
    obs_target_dim = 45
    num_samples = 100

    # Load dataset
    dataset, dataloader = load_dataset(
        dataset_path, obs_ee_dim, obs_shape_dim, obs_target_dim
    )

    # Extract shape data
    original_shapes, original_actions, processed_shapes, processed_actions = (
        extract_shapes(dataset_path, dataloader, obs_ee_dim, obs_shape_dim)
    )

    # Initialize processor
    dlo_processor = DloDataProcessor(processed_shapes)
    action_processor = ActionDataProcessor(
        processed_actions, num_points=15
    )

    # Randomly sample shapes for visualization
    indices = np.random.choice(
        len(original_shapes), size=min(num_samples, len(original_shapes)), replace=False
    )

    # Visualize normalization for selected shapes
    for i in indices:
        visualize_normalization(
            dlo_processor,
            action_processor,
            original_shapes[i].copy(),
            original_actions[i].copy(),
            i,
        )


if __name__ == "__main__":
    main()
