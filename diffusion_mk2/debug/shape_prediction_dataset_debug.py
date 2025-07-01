import zarr
import matplotlib.pyplot as plt
import numpy as np
from diffusion_mk2.dataset.shape_prediction_dataset import DloSample

np.set_printoptions(precision=4,    # number of decimal places
                    suppress=True,  # suppress scientific notation
                    linewidth=100,  # characters per line
                    threshold=1000) # controls summarization of large arrays

def plot_sample(ax, dlo_0, dlo_1, action, denormalize_idx=False, title=None):
    if not denormalize_idx:
        idx = int(action[0])
    else:
        idx = int(action[0] * (dlo_0.shape[0] - 1))


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

    print(f"Dataset size: {len(initial_shapes)}")
    num_points = initial_shapes.shape[1]
    print("Number of points:", num_points)

    sample_obj = DloSample(num_points=num_points, scale_action=False)

    for dlo_0, dlo_1, action in zip(initial_shapes, final_shapes, actions):
        dlo_0 = dlo_0[:, :2]
        dlo_1 = dlo_1[:, :2]

        if np.linalg.norm(dlo_0[0] - dlo_1[0]) > np.linalg.norm(dlo_0[0] - dlo_1[-1]):
            dlo_1 = np.flip(dlo_1, axis=0)

        dlo_0_n, dlo_1_n, action_n = sample_obj.normalize(dlo_0, dlo_1, action)
        dlo_0_dn, dlo_1_dn, action_dn = sample_obj.denormalize(dlo_0_n, dlo_1_n, action_n)

        # Print diagnostics
        print("Original action:", action)
        print("Normalized action:", action_n)
        print("Denormalized action:", action_dn)

        # Visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        plot_sample(ax1, dlo_0, dlo_1, action, title="Original")
        plot_sample(ax2, dlo_0_n, dlo_1_n, action_n, denormalize_idx=True, title="Normalized")
        plot_sample(ax3, dlo_0_dn, dlo_1_dn, action_dn, title="Denormalized")
        plt.tight_layout()
        plt.show()

