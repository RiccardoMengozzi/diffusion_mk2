import zarr, torch, random, json
import matplotlib.pyplot as plt
import numpy as np
from diffusion_mk2.dataset.pusht_state_dataset import PushTStateDataset
from diffusion_mk2.dataset.pusht_state_dataset import normalize_data, unnormalize_data, get_data_stats

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
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/combined_pushing_dataset.zarr.zip"
    obs_ee_dim = 5
    obs_shape_dim = 45
    obs_target_dim = 45
    act_dim = 5


    # dataset_root = zarr.open(dataset_path, 'r')

    # init_shape = dataset_root['data']['state'][:, obs_ee_dim :obs_ee_dim + obs_shape_dim].reshape(-1, 15, 3)
    # init_shape = init_shape[:, :, :2]  # Use only the first two dimensions for plotting
    # print(f"init_shape shape: {init_shape.shape}")

    json_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/json_data/filtered_combined_dataset.jsonl"
    init_shapes = []
    with open(json_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get("type") == "data":
                init_shapes.append(data["obs_dlo"])
    init_shapes = np.array(init_shapes, dtype=np.float32)


    indices = np.random.choice(len(init_shapes), size=20, replace=False)
    random_init_shapes = init_shapes[indices]

    for i,shape in enumerate(random_init_shapes):
        dlo_0 = shape.copy()
        dlo_0_n = normalize_data(dlo_0, get_data_stats(dlo_0))
        dlo_0_dn = unnormalize_data(dlo_0_n, get_data_stats(dlo_0))
        print("index:", indices[i])

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
