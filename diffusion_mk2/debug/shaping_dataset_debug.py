import zarr
import torch
import numpy as np
import math
from torch.utils.data import Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusion_mk2.model import normalization_pca
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
    subset_len = max(1, math.floor(1 * total_len))
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
    dlo_states = dlo_states.reshape(-1, num_points, 3)
    dlo_targets = dlo_targets.reshape(-1, num_points, 3)



    proc_ee = []
    proc_dlo = []
    proc_target = []
    proc_action = []
    for batch in tqdm(dataloader, desc="extracting data", total=len(dataloader)):
        proc_ee.append(batch["obs"][:, :, :obs_ee_dim].numpy().squeeze())
        proc_dlo.append(
            batch["obs"][:, :, obs_ee_dim : obs_ee_dim + obs_shape_dim]
            .numpy()
            .squeeze()
            .reshape(-1, obs_shape_dim // 3, 3)
        )
        proc_target.append(
            batch["obs"][:, :, obs_ee_dim + obs_shape_dim : obs_ee_dim + obs_shape_dim + obs_shape_dim]
            .numpy()
            .squeeze()
            .reshape(-1, obs_shape_dim // 3, 3)
        )

        proc_action.append(batch["action"].numpy().squeeze())

    proc_ee = np.array(proc_ee).squeeze()
    proc_dlo = np.array(proc_dlo).squeeze()
    proc_target = np.array(proc_target).squeeze()
    proc_action = np.array(proc_action).squeeze()

    return ee_states, dlo_states, dlo_targets, actions, proc_ee, proc_dlo, proc_target, proc_action



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
        interval=0.1,
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
        for i, (ee_states, dlo_states, target_states, action_states, title) in enumerate(datasets):
            ax = axes[i]
            ax.clear()

            # DLO 3D
            dlo_pts = dlo_states[frame_idx]
            ax.plot(
                dlo_pts[:, 0],
                dlo_pts[:, 1],
                dlo_pts[:, 2],
                "o-",
                label="DLO",
                linewidth=2,
                markersize=4,
            )
            target_pts = target_states[frame_idx]
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
            # ax.view_init(elev=90, azim=-90)

        plt.tight_layout()
        plt.pause(interval)

    plt.show()


def main():
    # Configuration
<<<<<<< HEAD:diffusion_mk2/debug/trajectory_pred_dataset_debug.py
    dataset_path = "/home/lar/Riccardo/diffusion_mk2/zarr_data/dataset.zarr.zip"
=======
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/dataset_cleaned_short.zarr.zip"
>>>>>>> b42eaf513f318d126281107cbc7675875456927c:diffusion_mk2/debug/shaping_dataset_debug.py
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


    ee_pos_norm = ee_norm[:, :3]  # [x, y, z]
    ee_theta_norm = ee_norm[:, 3]  # [theta]
    ee_gripper_norm = ee_norm[:, 4]  # [gripper]

    action_pos_norm = action_norm[:, :3]  # [x, y, z]
    action_theta_norm = action_norm[:, 3]  # [theta]
    action_gripper_norm = action_norm[:, 4]  # [gripper]

    # Denormalize normalized data
    ee_dn_arr = []
    dlo_dn_arr = []
    target_dn_arr = []
    action_dn_arr = []
    for dlo, ee_pos_n, ee_theta_n, ee_grip_n, dlo_n, target_n, act_pos_n, act_theta_n, act_grip_n in zip(
        dlo_orig,
        ee_pos_norm, 
        ee_theta_norm, 
        ee_gripper_norm, 
        dlo_norm, 
        target_norm, 
        action_pos_norm,
        action_theta_norm,
        action_gripper_norm
    ):
            cs0, csR = normalization_pca.compute_normalize_factors(dlo)
   
            ee_pos_dn = normalization_pca.denormalize_pca(ee_pos_n, cs0, csR)
            dlo_dn = normalization_pca.denormalize_pca(dlo_n, cs0, csR)
            target_dn = normalization_pca.denormalize_pca(target_n, cs0, csR)
            action_pos_dn = normalization_pca.denormalize_pca(act_pos_n, cs0, csR, rotation_only=True)
            ee_theta_dn = normalization_pca.denormalize_min_max(ee_theta_n, dataset.stats["obs_ee"]["min"][3], dataset.stats["obs_ee"]["max"][3])
            ee_gripper_dn = normalization_pca.denormalize_min_max(ee_grip_n, dataset.stats["obs_ee"]["min"][4], dataset.stats["obs_ee"]["max"][4])
            action_theta_dn = normalization_pca.denormalize_min_max(act_theta_n, dataset.stats["action"]["min"][3], dataset.stats["action"]["max"][3])
            action_gripper_dn = normalization_pca.denormalize_min_max(act_grip_n, dataset.stats["action"]["min"][4], dataset.stats["action"]["max"][4])

            ee_state_dn = np.concatenate([ee_pos_dn.squeeze(), np.array([ee_theta_dn]), np.array([ee_gripper_dn])])
            action_dn = np.concatenate([action_pos_dn, np.array([action_theta_dn]), np.array([action_gripper_dn])])

            ee_dn_arr.append(ee_state_dn)
            dlo_dn_arr.append(dlo_dn)
            target_dn_arr.append(target_dn)
            action_dn_arr.append(action_dn)



    ee_dn_arr = np.array(ee_dn_arr)
    dlo_dn_arr = np.array(dlo_dn_arr)
    target_dn_arr = np.array(target_dn_arr)
    action_dn_arr = np.array(action_dn_arr)

    # Print some statistics
    print(f"Number of frames: {len(ee_orig)}")
    print(f"EE state shape: {ee_orig.shape}")
    print(f"DLO state shape: {dlo_orig.shape}")
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
        ee_dn_arr,
        dlo_dn_arr,
        target_dn_arr,
        action_dn_arr,
        interval=0.1,
    )

    # Keep the script running to see all plots
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
