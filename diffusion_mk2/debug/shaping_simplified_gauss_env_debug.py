import zarr
import torch
import numpy as np
import math
import os
from torch.utils.data import Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusion_mk2.model import normalization_pca
from diffusion_mk2.dataset.shaping_dataset import ShapingDataset
from diffusion_mk2.inference.shaping_simplified_inference_class import ShapingInference
import collections


np.set_printoptions(precision=4,    # number of decimal places
                    suppress=True,  # suppress scientific notation
                    linewidth=100,  # characters per line
                    threshold=1000) # controls summarization of large arrays


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



import numpy as np
import matplotlib.pyplot as plt
import collections

def plot(
    ee_orig,
    dlo_orig,
    target_orig,
    action_orig,
    model,
    interval=0.1,
):
    """Plot initial shape, target shape, ideal action and predicted action."""

    fig, ax = plt.subplots(figsize=(6, 6))

    # Fixed axis limits
    x_min, x_max = 0.4, 0.6
    y_min, y_max = -0.1, 0.1

    num_frames = min(len(ee_orig), len(dlo_orig))

    # Prepare observation history
    obs = np.concatenate(
        [ee_orig[0], dlo_orig[0].flatten(), target_orig[0].flatten()],
        axis=-1
    ).flatten()

    obs_horizon = model.obs_horizon
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    for frame_idx in range(num_frames):
        ax.clear()

        ee_states = ee_orig[frame_idx]
        dlo_states = dlo_orig[frame_idx]
        target_states = target_orig[frame_idx]
        action_states = action_orig[frame_idx]

        # Plot current DLO
        ax.plot(dlo_states[:, 0], dlo_states[:, 1], "o-", label="DLO", linewidth=2, markersize=4)

        # Plot target shape
        ax.plot(target_states[:, 0], target_states[:, 1], "o-", label="Target", linewidth=2, markersize=4)

        # Plot end-effector position
        ee = ee_states[:2]
        ax.scatter([ee[0]], [ee[1]], marker="o", s=60, c="g", label="End Effector")

        #### INFERENCE ####
        obs = np.concatenate(
            [ee_states, dlo_states.flatten(), target_states.flatten()],
            axis=-1
        ).flatten()

        obs_deque.append(obs)
        obs_stack = np.stack(obs_deque).reshape(model.obs_horizon, -1)

        pred_action, _ = model.run_inference(observation=obs_stack)

        pred_idx = int(pred_action[0, 0])
        pred_delta = pred_action[:, 1:3]  # Take only first prediction if batch

        # Compute normalized direction
        ideal_delta = action_states[:2]
        norm = np.linalg.norm(ideal_delta)
        if norm == 0:
            direction = np.zeros(2)
        else:
            direction = ideal_delta / norm

        # Scale to fixed arrow length
        arrow = direction * 0.02

        # Plot 2D arrow from ee
        ax.arrow(
            ee[0], ee[1],             # Start at (x, y)
            arrow[0], arrow[1],       # dx, dy
            head_width=0.003,
            head_length=0.005,
            fc='r',
            ec='r',
            linewidth=1.5,
            label="Ideal Action"
        )

        # Plot predicted action
        pred_pt = []
        print("pred_idx", pred_idx)
        pt = dlo_states[pred_idx, :2].copy()  # Start from predicted DLO point
        for delta in pred_delta:
            pt += delta
            pred_pt.append(pt.copy())  
        pred_pt = np.array(pred_pt)
        ax.plot(pred_pt[:, 0], pred_pt[:, 1], "^-", label="Predicted Action", linewidth=2, markersize=4)

        # Set axes
        ax.set_title(f"Frame {frame_idx + 1}/{num_frames} - DLO and Target")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

        # Add legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper left")

        plt.pause(interval)

    plt.show()


def main():
    # Configuration
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(project_dir, "zarr_data", "simplified_short.zarr.zip")
    
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

    ee_orig_simplified = ee_orig[:, [0, 1, 3]]
    dlo_orig_simplified = dlo_orig[:, :, :2]
    target_orig_simplified = target_orig[:, :, :2]
    action_orig_simplified = action_orig[:, [0, 1, 3]]

    model = ShapingInference(
        ckp_path= os.path.join(project_dir, "weights", "chkp_resilient-armadillo-50_epoch_7400.pt"),
        num_timesteps=10,
        verbose=False
    )


    plot(
        ee_orig_simplified,
        dlo_orig_simplified,
        target_orig_simplified,
        action_orig_simplified,
        model,
        interval=0.1,
    )

    # Keep the script running to see all plots
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
