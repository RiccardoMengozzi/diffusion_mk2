import zarr
import torch
import numpy as np
import math
from torch.utils.data import Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusion_mk2.model.normalization_simple import denormalize_data
from diffusion_mk2.dataset.shaping_dataset import ShapingDataset 

# Configure numpy printing
np.set_printoptions(precision=8, suppress=True, linewidth=100, threshold=1000)


def load_dataset(dataset_path, obs_ee_dim, obs_shape_dim, obs_target_dim):
    """Load and return dataset and dataloader."""
    dataset = ShapingDataset(
        dataset_path=dataset_path,
        pred_horizon=1,
        obs_horizon=1,
        action_horizon=1,
        obs_ee_dim=obs_ee_dim,
        obs_dlo_dim=obs_shape_dim,
        obs_target_dim=obs_target_dim,
    )

    total_len = len(dataset)
    subset_len = max(1, math.floor(0.01 * total_len))
    subset_indices = list(range(subset_len))

    # 3) build a Subset and DataLoader over that
    small_ds = Subset(dataset, subset_indices)

    dataloader = torch.utils.data.DataLoader(
        small_ds,
        batch_size=1,
        num_workers=12,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataset, dataloader


def extract_all_states(dataset_path, dataloader, obs_ee_dim, obs_shape_dim):
    """Extract full sequences of original and processed DLO and EE states."""
    root = zarr.open(dataset_path, mode="r")
    states = root["data"]["state"]
    actions = root["data"]["action"]
    ee_states = states[:, :obs_ee_dim]  # [x, y, z, θ, grip]
    dlo_states = states[:, obs_ee_dim : obs_ee_dim + obs_shape_dim]
    dlo_targets = states[:, obs_ee_dim + obs_shape_dim : obs_ee_dim + obs_shape_dim + obs_shape_dim]
    num_points = obs_shape_dim // 3
    dlo_states = dlo_states.reshape(-1, num_points, obs_shape_dim // num_points)
    dlo_targets = dlo_targets.reshape(-1, num_points, obs_shape_dim // num_points)

    norm_ee = []
    norm_dlo = []
    norm_dlo_target = []
    norm_action = []
    for batch in tqdm(dataloader, desc="extracting data", total=len(dataloader)):
        norm_ee.append(batch["obs"][:, :, :obs_ee_dim].numpy().squeeze())
        norm_dlo.append(
            batch["obs"][:, :, obs_ee_dim : obs_ee_dim + obs_shape_dim]
            .numpy()
            .squeeze()
            .reshape(-1, obs_shape_dim // 3, 3)
        )
        norm_dlo_target.append(
            batch["obs"][:, :, obs_ee_dim + obs_shape_dim : obs_ee_dim + 2 * obs_shape_dim]
            .numpy()
            .squeeze()
            .reshape(-1, obs_shape_dim // 3, 3)
        )
        norm_action.append(batch["action"].numpy().squeeze())

    norm_ee = np.array(norm_ee).squeeze()
    norm_dlo = np.array(norm_dlo).squeeze()
    norm_dlo_target = np.array(norm_dlo_target).squeeze()
    norm_action = np.array(norm_action).squeeze()

    return ee_states, dlo_states, dlo_targets, actions, norm_ee, norm_dlo, norm_dlo_target, norm_action



def plot_animated_comparison(
    ee_orig,
    dlo_orig,
    target_orig,
    action_orig,
    ee_norm,
    dlo_norm,
    target_norm,
    action_norm,
    ee_dn,
    dlo_dn,
    target_dn,
    action_dn,
    interval=0.5,
):
    """Plot animated comparison of all three sequences side by side."""
    fig = plt.figure(figsize=(18, 6))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    datasets = [
        (ee_orig, dlo_orig, target_orig, action_orig, "Original"),
        (ee_norm, dlo_norm, target_norm, action_norm, "Normalized"),
        (ee_dn, dlo_dn, target_dn, action_dn, "Denormalized"),
    ]

    # Precompute axis limits for each dataset
    limits = []
    x_min, x_max = 0.4, 0.6
    y_min, y_max = -0.1, 0.1
    z_min, z_max = 0.6, 0.8
    limits.append((x_min, x_max, y_min, y_max, z_min, z_max))  # Placeholder for limits
    x_min, x_max = 0.2, 0.8
    y_min, y_max = -0.3, 0.3
    z_min, z_max = -0.6, 0.6
    limits.append((x_min, x_max, y_min, y_max, z_min, z_max))  # Placeholder for limits
    x_min, x_max = 0.4, 0.6
    y_min, y_max = -0.1, 0.1
    z_min, z_max = 0.6, 0.8
    limits.append((x_min, x_max, y_min, y_max, z_min, z_max))  # Placeholder for limits
    # for ee_states, dlo_states, target_states, action_states, title in datasets:
    #     all_pts = dlo_states.reshape(-1, 3)
    #     x_min, y_min, z_min = all_pts[:,0].min(), all_pts[:,1].min(), all_pts[:,2].min()
    #     x_max, y_max, z_max = all_pts[:,0].max(), all_pts[:,1].max(), all_pts[:,2].max()
    #     limits.append((x_min, x_max, y_min, y_max, z_min, z_max))

    num_frames = min(len(ee_orig), len(dlo_orig))


    for frame_idx in range(num_frames):
        for i, (ee_states, dlo_states, target_states, action_states, title) in enumerate(datasets):
            ax = axes[i]
            ax.clear()

            # DLO 3D
            dlo_pts = dlo_states[frame_idx]
            target_pts = target_states[frame_idx]
            ax.plot(
                dlo_pts[:, 0],
                dlo_pts[:, 1],
                dlo_pts[:, 2],
                "o-",
                label="DLO",
                linewidth=2,
                markersize=4,
            )
            ax.plot(
                target_pts[:, 0],
                target_pts[:, 1],
                target_pts[:, 2],
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
            if title == "Normalized":
                action_pt = delta + ee
            else:
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
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/dataset.zarr.zip"
    obs_ee_dim = 5  # [x, y, z, θ, grip]
    obs_shape_dim = 45  # 15 points * 3
    obs_target_dim = 45



    # Load and extract
    dataset, dataloader = load_dataset(
        dataset_path, obs_ee_dim, obs_shape_dim, obs_target_dim
    )
    ee_orig, dlo_orig, target_orig, action_orig, ee_norm, dlo_norm, target_norm, action_norm = extract_all_states(
        dataset_path, dataloader, obs_ee_dim, obs_shape_dim
    )
    ee_orig = ee_orig[:len(dataloader), :]
    dlo_orig = dlo_orig[:len(dataloader), :, :]
    target_orig = target_orig[:len(dataloader), :, :]
    action_orig = action_orig[:len(dataloader), :]

    stats = dataset.stats
    # Denormalize normalized data
    dlo_dn = []
    target_dn = []
    ee_dn = []
    action_dn = []
    for ee_n, dlo_n, target_n, action_n in zip(ee_norm, dlo_norm, target_norm, action_norm):
        ee_spatial_n = ee_n[:3]  # [x, y, z]
        ee_theta_n = ee_n[3]  # [theta]
        ee_grip_n = ee_n[4]  # [gripper]

        ee_spatial_dn = denormalize_data(ee_spatial_n, stats["obs_ee_spatial"], keep_zero_centered=False)
        ee_theta_dn = denormalize_data(ee_theta_n, stats["obs_ee_theta"], keep_zero_centered=False)
        ee_grip_dn = denormalize_data(ee_grip_n, stats["obs_ee_gripper"], keep_zero_centered=False)

        ee_dn.append(np.concatenate([ee_spatial_dn, ee_theta_dn, ee_grip_dn]))
        dlo_dn.append(denormalize_data(dlo_n, stats["obs_dlo"], keep_zero_centered=False))
        target_dn.append(denormalize_data(target_n, stats["obs_target"], keep_zero_centered=False))
        action_dn.append(denormalize_data(action_n, stats["action"], keep_zero_centered=True))
    dlo_dn = np.array(dlo_dn)
    ee_dn = np.array(ee_dn)
    target_dn = np.array(target_dn)
    action_dn = np.array(action_dn)
    # # stampa min e max per ciascuna delle tre coordinate dello stato DLO normalizzato
    # print("min/max normalized dlo x:", np.min(dlo_norm[:, :, 0]), np.max(dlo_norm[:, :, 0]))
    # print("min/max normalized dlo y:", np.min(dlo_norm[:, :, 1]), np.max(dlo_norm[:, :, 1]))
    # print("min/max normalized dlo z:", np.min(dlo_norm[:, :, 2]), np.max(dlo_norm[:, :, 2]))

    # # stessa cosa per l’unnormalized
    # print("min/max unnormalized dlo x:", np.min(dlo_dn[:, :, 0]), np.max(dlo_dn[:, :, 0]))
    # print("min/max unnormalized dlo y:", np.min(dlo_dn[:, :, 1]), np.max(dlo_dn[:, :, 1]))
    # print("min/max unnormalized dlo z:", np.min(dlo_dn[:, :, 2]), np.max(dlo_dn[:, :, 2]))

    # # per ee_state, che immagino sia un vettore 1D o 2D:
    # print("min/max normalized ee_state x:", np.min(ee_norm[:, 0]), np.max(ee_norm[:, 0]))
    # print("min/max normalized ee_state y:", np.min(ee_norm[:, 1]), np.max(ee_norm[:, 1]))
    # print("min/max normalized ee_state z:", np.min(ee_norm[:, 2]), np.max(ee_norm[:, 2]))

    # print("min/max unnormalized ee_state x:", np.min(ee_dn[:, 0]), np.max(ee_dn[:, 0]))
    # print("min/max unnormalized ee_state y:", np.min(ee_dn[:, 1]), np.max(ee_dn[:, 1]))
    # print("min/max unnormalized ee_state z:", np.min(ee_dn[:, 2]), np.max(ee_dn[:, 2]))



    # Print some statistics
    print(f"Number of frames: {len(ee_orig)}")
    print(f"EE state shape: {ee_orig.shape}")
    print(f"DLO state shape: {dlo_orig.shape}")
    print(f"Target state shape: {target_orig.shape}")
    print(f"Action shape: {action_orig.shape}")


    plot_animated_comparison(
        ee_orig,
        dlo_orig,
        target_orig,
        action_orig,
        ee_norm,
        dlo_norm,
        target_norm,
        action_norm,
        ee_dn,
        dlo_dn,
        target_dn,
        action_dn,
        interval=0.1,
    )

    # Keep the script running to see all plots
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
