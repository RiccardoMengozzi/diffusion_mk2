import zarr
import matplotlib.pyplot as plt
import numpy as np
from diffusion_mk2.model import normalization_pca
from diffusion_mk2.dataset.shape_prediction_dataset import DloDataset
import torch
import os
from tqdm import tqdm

np.set_printoptions(precision=4,    # number of decimal places
                    suppress=True,  # suppress scientific notation
                    linewidth=100,  # characters per line
                    threshold=1000) # controls summarization of large arrays


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_sample(ax, dlo_0, dlo_1, action, denormalize_idx=False, dataset=None, title=None):
    if not denormalize_idx:
        idx = int(action[0])
    else:
        idx = normalization_pca.denormalize_min_max(
            action[0], 
            dataset.stats["action"]["min"][0], 
            dataset.stats["action"]["max"][0]
        )
        idx = int(idx)

    init_idx_pos = dlo_0[idx]
    target_idx_pos = dlo_1[idx]

    ax.plot(dlo_0[:, 0], dlo_0[:, 1], "o-", label="dlo_0")
    ax.plot(dlo_1[:, 0], dlo_1[:, 1], "o-", label="dlo_1")
    ax.plot(init_idx_pos[0], init_idx_pos[1], "o-", label="grasp")
    ax.plot(target_idx_pos[0], target_idx_pos[1], "o-", label="release")

    ax.arrow(init_idx_pos[0], 
             init_idx_pos[1], 
             action[1], 
             action[2], 
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


def extract_all_states(dataset_path, dataset):
    """Extract full sequences of original and processed DLO and EE states."""
    root = zarr.open(dataset_path, mode="r")
    inits = root["data"]["initial_shape"][:].reshape(-1, dataset.num_points, 3)
    targets = root["data"]["final_shape"][:].reshape(-1, dataset.num_points, 3)
    actions = root["data"]["action"][:]

    inits = inits[:, :, :2]
    targets = targets[:, :, :2]

    normalized_data = np.array(dataset.normalized_train_data)
    proc_inits = np.array(list(map(lambda d: d.get("initial_shape"), normalized_data)))
    proc_targets = np.array(list(map(lambda d: d.get("final_shape"), normalized_data)))
    proc_actions = np.array(list(map(lambda d: d.get("action"), normalized_data)))

    return inits, targets, actions, proc_inits, proc_targets, proc_actions

def load_dataset(dataset_path, num_points):
    """Load and return dataset and dataloader."""
    dataset = DloDataset(
        dataset_path=dataset_path,
        num_points=num_points,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=12,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataset, dataloader


if __name__ == "__main__":
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/shape_prediction.zarr.zip"
    
    dataset, dataloader = load_dataset(
        dataset_path=dataset_path,
        num_points=15,
    )

    inits, targets, actions, inits_n, targets_n, actions_n = extract_all_states(
        dataset_path, dataset
    )
    print("action_n shape:", actions_n.shape)

    actions_n_idx = actions_n[:, 0]
    actions_n_pos = actions_n[:, 1:3]  # Assuming the first 3 columns are the action positions
    actions_n_theta = actions_n[:, 3]

    for i, (dlo_0, dlo_1, action, dlo_0_n, dlo_1_n, action_n_idx, action_n_pos, action_n_theta) in enumerate(zip(inits,
                                                                                                  targets,
                                                                                                  actions,
                                                                                                  inits_n, 
                                                                                                  targets_n, 
                                                                                                  actions_n_idx,
                                                                                                  actions_n_pos,
                                                                                                  actions_n_theta)):
        cs0, csR = normalization_pca.compute_normalize_factors(dlo_0)
        print("dlo_0_n shape:", dlo_0_n.shape)
        dlo_0_dn = normalization_pca.denormalize_pca(dlo_0_n, cs0, csR)
        dlo_1_dn = normalization_pca.denormalize_pca(dlo_1_n, cs0, csR)
        action_dn_idx = normalization_pca.denormalize_min_max(action_n_idx, 
                                                     dataset.stats["action"]["min"][0],
                                                     dataset.stats["action"]["max"][0])
        action_dn_pos = normalization_pca.denormalize_pca(action_n_pos, cs0, csR, rotation_only=True)
        action_dn_theta = normalization_pca.denormalize_min_max(action_n_theta,
                                                                dataset.stats["action"]["min"][-1],
                                                                dataset.stats["action"]["max"][-1])
        
        action_n = np.array([action_n_idx, action_n_pos[0], action_n_pos[1], action_n_theta])
        action_dn = np.array([action_dn_idx, action_dn_pos[0], action_dn_pos[1], action_dn_theta])


        # Print diagnostics
        print("Original action:", action)
        print("Normalized action:", action_n)
        print("Denormalized action:", action_dn)

        # Visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        plot_sample(ax1, dlo_0, dlo_1, action, title=f"Original_{i}")
        plot_sample(ax2, dlo_0_n, dlo_1_n, action_n, denormalize_idx=True, dataset=dataset, title=f"Normalized_{i}")
        plot_sample(ax3, dlo_0_dn, dlo_1_dn, action_dn, title=f"Denormalized_{i}")
        plt.tight_layout()
        plt.show()

