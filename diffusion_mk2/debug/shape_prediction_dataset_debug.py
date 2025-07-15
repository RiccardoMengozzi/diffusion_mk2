import zarr
import matplotlib.pyplot as plt
import numpy as np
from diffusion_mk2.model import normalization_pca
from diffusion_mk2.dataset.shape_prediction_dataset import DloDataset

np.set_printoptions(precision=4,    # number of decimal places
                    suppress=True,  # suppress scientific notation
                    linewidth=100,  # characters per line
                    threshold=1000) # controls summarization of large arrays

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


if __name__ == "__main__":
    dataset_path = "/home/lar/Riccardo/diffusion_mk2/zarr_data/shape_prediction.zarr.zip"
    dataset_root = zarr.open(dataset_path, 'r')
    initial_shapes = dataset_root['data']['initial_shape'][:].reshape(-1, 15, 3)
    final_shapes = dataset_root['data']['final_shape'][:].reshape(-1, 15, 3)
    actions = dataset_root['data']['action'][:]

    dataset = DloDataset(dataset_path, num_points=15)

    actions_idx = actions[:, 0]
    actions_pos = actions[:, 1:3]  # Assuming the first 3 columns are the action positions
    actions_theta = actions[:, 3]

    initial_shapes = initial_shapes[:, :, :2]  # Use only the first two dimensions for plotting
    final_shapes = final_shapes[:, :, :2]  # Use only the first two dimensions


    for dlo_0, dlo_1, action_idx, action_pos, action_theta in zip(initial_shapes, 
                                                                  final_shapes, 
                                                                  actions_idx,
                                                                  actions_pos,
                                                                  actions_theta):
        
        cs0, csR = normalization_pca.compute_normalize_factors(dlo_0)
        dlo_0_n = normalization_pca.normalize_pca(dlo_0, cs0, csR)
        dlo_1_n = normalization_pca.normalize_pca(dlo_1, cs0, csR)
        action_idx_n = normalization_pca.normalize_min_max(action_idx, 
                                                           dataset.stats["action"]["min"][0],
                                                           dataset.stats["action"]["max"][0])
        action_pos_n = normalization_pca.normalize_pca(action_pos, cs0, csR, rotation_only=True)
        action_theta_n = normalization_pca.normalize_min_max(action_theta,
                                                           dataset.stats["action"]["min"][-1],
                                                           dataset.stats["action"]["max"][-1]) 


        dlo_0_dn = normalization_pca.denormalize_pca(dlo_0_n, cs0, csR)
        dlo_1_dn = normalization_pca.denormalize_pca(dlo_1_n, cs0, csR)
        action_idx_dn = normalization_pca.denormalize_min_max(action_idx_n, 
                                                     dataset.stats["action"]["min"][0],
                                                     dataset.stats["action"]["max"][0])
        action_pos_dn = normalization_pca.denormalize_pca(action_pos_n, cs0, csR, rotation_only=True)
        action_theta_dn = normalization_pca.denormalize_min_max(action_theta_n,
                                                                dataset.stats["action"]["min"][-1],
                                                                dataset.stats["action"]["max"][-1])
        
        action = np.array([action_idx, action_pos[0], action_pos[1], action_theta])
        action_n = np.array([action_idx_n, action_pos_n[0], action_pos_n[1], action_theta_n])
        action_dn = np.array([action_idx_dn, action_pos_dn[0], action_pos_dn[1], action_theta_dn])


        # Print diagnostics
        print("Original action:", action)
        print("Normalized action:", action_n)
        print("Denormalized action:", action_dn)

        # Visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        plot_sample(ax1, dlo_0, dlo_1, action, title="Original")
        plot_sample(ax2, dlo_0_n, dlo_1_n, action_n, denormalize_idx=True, dataset=dataset, title="Normalized")
        plot_sample(ax3, dlo_0_dn, dlo_1_dn, action_dn, title="Denormalized")
        plt.tight_layout()
        plt.show()

