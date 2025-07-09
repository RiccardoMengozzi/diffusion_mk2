import zarr
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusion_mk2.model.normalization import (
    DloDataProcessor,
    EEStateDataProcessor,
    ActionDataProcessor,
)
from diffusion_mk2.dataset.pusht_state_dataset import PushTStateDataset

# Configure numpy printing
np.set_printoptions(precision=8, suppress=True, linewidth=100, threshold=1000)


def load_dataset(dataset_path, obs_ee_dim, obs_shape_dim, obs_target_dim):
    """Load and return dataset and dataloader."""
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
        num_workers=10,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader


def extract_all_states(dataset_path, dataloader, obs_ee_dim, obs_shape_dim):
    """Extract full sequences of original and processed DLO and EE states."""
    root = zarr.open(dataset_path, mode="r")
    states = root["data"]["state"]
    actions = root["data"]["action"]
    ee_states = states[:, :obs_ee_dim]  # [x, y, z, θ, grip]
    dlo_states = states[:, obs_ee_dim : obs_ee_dim + obs_shape_dim]
    num_points = obs_shape_dim // 3
    dlo_states = dlo_states.reshape(-1, num_points, 3)

    proc_ee = []
    proc_dlo = []
    proc_action = []
    for batch in tqdm(dataloader, desc="extracting data", total=len(dataloader)):
        proc_ee.append(batch["obs"][:, :, :obs_ee_dim].numpy().squeeze())
        proc_dlo.append(
            batch["obs"][:, :, obs_ee_dim : obs_ee_dim + obs_shape_dim]
            .numpy()
            .squeeze()
            .reshape(-1, obs_shape_dim // 3, 3)
        )
        proc_action.append(batch["action"].numpy().squeeze())

    proc_ee = np.array(proc_ee).squeeze()
    proc_dlo = np.array(proc_dlo).squeeze()
    proc_action = np.array(proc_action).squeeze()

    return ee_states, dlo_states, actions, proc_ee, proc_dlo, proc_action



def plot_animated_comparison(
    ee_orig,
    dlo_orig,
    action_orig,
    ee_norm,
    dlo_norm,
    action_norm,
    ee_dn,
    dlo_dn,
    action_dn,
    interval=0.5,
):
    """Plot animated comparison of all three sequences side by side."""
    fig = plt.figure(figsize=(18, 6))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    datasets = [
        (ee_orig, dlo_orig, action_orig, "Original"),
        (ee_norm, dlo_norm, action_norm, "Normalized"),
        (ee_dn, dlo_dn, action_dn, "Denormalized"),
    ]

    # Precompute axis limits for each dataset
    limits = []
    x_min, x_max = 0.4, 0.6
    y_min, y_max = -0.1, 0.1
    z_min, z_max = 0.6, 0.8
    limits.append((x_min, x_max, y_min, y_max, z_min, z_max))  # Placeholder for limits
    x_min, x_max = -0.1, 0.1
    y_min, y_max = -0.1, 0.1
    z_min, z_max = -0.1, 0.1
    limits.append((x_min, x_max, y_min, y_max, z_min, z_max))  # Placeholder for limits
    x_min, x_max = 0.4, 0.6
    y_min, y_max = -0.1, 0.1
    z_min, z_max = 0.6, 0.8
    limits.append((x_min, x_max, y_min, y_max, z_min, z_max))  # Placeholder for limits
    # for ee_states, dlo_states, title in datasets:
    #     all_pts = dlo_states.reshape(-1, 3)
    #     x_min, y_min = all_pts[:,0].min(), all_pts[:,1].min()
    #     x_max, y_max = all_pts[:,0].max(), all_pts[:,1].max()
    #     limits.append((x_min, x_max, y_min, y_max))

    num_frames = min(len(ee_orig), len(dlo_orig))


    for frame_idx in range(num_frames):
        for i, (ee_states, dlo_states, action_states, title) in enumerate(datasets):
            ax = axes[i]
            ax.clear()

            # DLO 3D
            pts = dlo_states[frame_idx]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                "o-",
                label="DLO",
                linewidth=2,
                markersize=4,
            )

            # End‑Effector 3D
            ee = ee_states[frame_idx][:3]

            ax.scatter(
                [ee[0]], [ee[1]], [ee[2]], marker="o", s=60, c="g", label="End Effector"
            )

            # Action target 3D (EE + delta)
            delta = action_states[frame_idx][:3]
            action_pt = ee + delta
            ax.scatter(
                [action_pt[0]],
                [action_pt[1]],
                [action_pt[2]],
                marker="*",
                s=80,
                c="r",
                label="Action",
            )

            ax.set_title(f"{title} Frame {frame_idx}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            # Imposta limiti 3D coerenti
            x_min, x_max, y_min, y_max, z_min, z_max = limits[i]
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            # Vista top‑down: elevazione 90°, rotazione orizzontale  -90° (opzionale)
            ax.view_init(elev=90, azim=-90)

        plt.tight_layout()
        plt.pause(interval)

    plt.show()


def main():
    # Configuration
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/dataset_fixed.zarr.zip"
    obs_ee_dim = 5  # [x, y, z, θ, grip]
    obs_shape_dim = 45  # 15 points * 3
    obs_target_dim = 45

    # Load and extract
    dataloader = load_dataset(
        dataset_path, obs_ee_dim, obs_shape_dim, obs_target_dim
    )
    ee_orig, dlo_orig, action_orig, ee_norm, dlo_norm, action_norm = extract_all_states(
        dataset_path, dataloader, obs_ee_dim, obs_shape_dim
    )

    ee_states_processor : EEStateDataProcessor = dataloader.dataset.ee_states_processor
    actions_processor : ActionDataProcessor = dataloader.dataset.actions_processor
    dlo_processor : DloDataProcessor = dataloader.dataset.initial_shapes_processor

    ## Descale normalized data for coherent visualization: 
    for i in range(len(ee_norm)):
        ee_norm[i] = ee_states_processor.descale(ee_norm[i])
        action_norm[i] = actions_processor.descale(action_norm[i])

    # Denormalize normalized data
    dlo_dn = []
    ee_dn = []
    action_dn = []
    for i, (ee_n, dlo_n, action_n) in enumerate(zip(ee_norm, dlo_norm, action_norm)):
        # REMEMBER descale=False here as I already did it above
        ee_dn.append(ee_states_processor.denormalize_sample(ee_n, descale=False, idx=i))
        dlo_dn.append(dlo_processor.denormalize_sample(dlo_n, idx=i))
        action_dn.append(actions_processor.denormalize_sample(action_n, descale=False, idx=i))
    dlo_dn = np.array(dlo_dn)
    ee_dn = np.array(ee_dn)
    action_dn = np.array(action_dn)

    # Print some statistics
    print(f"Number of frames: {len(ee_orig)}")
    print(f"EE state shape: {ee_orig.shape}")
    print(f"DLO state shape: {dlo_orig.shape}")
    print(f"Action shape: {action_orig.shape}")


    plot_animated_comparison(
        ee_orig,
        dlo_orig,
        action_orig,
        ee_norm,
        dlo_norm,
        action_norm,
        ee_dn,
        dlo_dn,
        action_dn,
        interval=0.1,
    )

    # Keep the script running to see all plots
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
