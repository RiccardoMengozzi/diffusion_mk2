import zarr, torch, random, json
import matplotlib.pyplot as plt
import numpy as np
from diffusion_mk2.model.normalization import DloDataProcessor, ActionDataProcessor

np.set_printoptions(precision=8,    # number of decimal places
                    suppress=True,  # suppress scientific notation
                    linewidth=100,  # characters per line
                    threshold=1000) # controls summarization of large arrays

def plot_shape(ax, dlo_0, title=None):



    ax.plot(dlo_0[:, 0], dlo_0[:, 1], "o-", label="dlo_0")


    if title:
        ax.set_title(title)

    ax.legend()
    ax.axis("equal")


if __name__ == "__main__":
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/pushing_dataset_better.zip"
    obs_ee_dim = 5
    obs_shape_dim = 45
    obs_target_dim = 45
    act_dim = 5


    dataset_root = zarr.open(dataset_path, 'r')

    init_shapes = dataset_root['data']['state'][:, obs_ee_dim :obs_ee_dim + obs_shape_dim].reshape(-1, 15, 3)
    init_shapes = init_shapes[:, :, :2]  # Use only the first two dimensions for plotting
    print(f"init_shape shape: {init_shapes.shape}")

    init_shapes = np.array(init_shapes, dtype=np.float32)
    dlo_processor = DloDataProcessor(init_shapes)


    indices = np.random.choice(len(init_shapes), size=20, replace=False)
    random_init_shapes = init_shapes[indices]

    for i,shape in enumerate(random_init_shapes):
        dlo_0 = shape.copy()
        cs0, csR = dlo_processor._compute_normalize_factors(dlo_0)
        dlo_0_n = dlo_processor.normalize(dlo_0, cs0, csR)
        dlo_0_dn = dlo_processor.denormalize(dlo_0_n, cs0, csR)

        print("Original shape:\n", dlo_0)
        print("Normalized shape:\n", dlo_0_n)
        print("Denormalized shape:\n", dlo_0_dn)

        # Visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        plot_shape(ax1, dlo_0, title="Original")
        plot_shape(ax2, dlo_0_n, title="Normalized")
        plot_shape(ax3, dlo_0_dn, title="Denormalized")
        plt.tight_layout()
        plt.show()
