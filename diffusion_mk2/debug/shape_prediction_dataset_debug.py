import zarr
import matplotlib.pyplot as plt
import numpy as np
from diffusion_mk2.model.normalization import DloDataProcessor, ActionDataProcessor

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
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/shape_pred_test_first.zarr.zip"
    dataset_root = zarr.open(dataset_path, 'r')
    initial_shapes = dataset_root['data']['initial_shape'][:].reshape(-1, 15, 3)
    final_shapes = dataset_root['data']['final_shape'][:].reshape(-1, 15, 3)
    actions = dataset_root['data']['action'][:]

    initial_shapes = initial_shapes[:, :, :2]  # Use only the first two dimensions for plotting
    final_shapes = final_shapes[:, :, :2]  # Use only the first two dimensions

    dlo_0_processor = DloDataProcessor(initial_shapes)
    dlo_1_processor = DloDataProcessor(final_shapes)
    action_processor = ActionDataProcessor(actions, num_points=15)


    for dlo_0, dlo_1, action in zip(initial_shapes, final_shapes, actions):
        
        cs0, csR = dlo_0_processor._compute_normalize_factors(dlo_0)
        dlo_0_n = dlo_0_processor.normalize(dlo_0, cs0, csR)
        dlo_1_n = dlo_1_processor.normalize(dlo_1, cs0, csR)
        action_n = action_processor.normalize(action, cs0, csR)

        dlo_0_dn = dlo_0_processor.denormalize(dlo_0_n, cs0, csR)
        dlo_1_dn = dlo_1_processor.denormalize(dlo_1_n, cs0, csR)
        action_dn = action_processor.denormalize(action_n, cs0, csR)


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

