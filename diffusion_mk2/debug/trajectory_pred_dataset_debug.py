import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusion_mk2.model.normalization import DloDataProcessor, EEStateDataProcessor, ActionDataProcessor
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
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataset, dataloader


def extract_all_states(dataset_path, dataloader, obs_ee_dim, obs_shape_dim):
    """Extract full sequences of original and processed DLO and EE states."""
    root = zarr.open(dataset_path, mode='r')
    states = root['data']['state']
    actions = root['data']['action']
    ee_states = states[:, :obs_ee_dim]      # [x, y, z, θ, grip]
    dlo_states = states[:, obs_ee_dim:obs_ee_dim+obs_shape_dim]
    num_points = obs_shape_dim // 3
    dlo_states = dlo_states.reshape(-1, num_points, 3)


    proc_ee = []
    proc_dlo = []
    proc_action = []
    for batch in dataloader:
        proc_ee.append(batch['obs'][:, :, :obs_ee_dim].numpy().squeeze())
        proc_dlo.append(batch['obs'][:, :, obs_ee_dim:obs_ee_dim+obs_shape_dim].numpy().squeeze().reshape(-1, obs_shape_dim // 3, 3))
        proc_action.append(batch['action'].numpy().squeeze())

    proc_ee = np.array(proc_ee).squeeze()
    proc_dlo = np.array(proc_dlo).squeeze()
    proc_action = np.array(proc_action).squeeze()

    return ee_states, dlo_states, actions, proc_ee, proc_dlo, proc_action


def plot_sequence_xy(ee_states, dlo_states, title_prefix, interval=0.1):
    """Plot DLO and EE states in XY plane sequentially with title prefix."""
    fig, ax = plt.subplots(figsize=(6,6))

    # Precompute axis limits
    all_pts = dlo_states.reshape(-1, 3)
    x_min, y_min = all_pts[:,0].min(), all_pts[:,1].min()
    x_max, y_max = all_pts[:,0].max(), all_pts[:,1].max()
    print(dlo_states.shape)
    for i in range(dlo_states.shape[0]):
        ax.clear()
        pts = dlo_states[i]
        ax.plot(pts[:,0], pts[:,1], 'o-', label='DLO')
        ee = ee_states[i][:2]
        ax.scatter([ee[0]], [ee[1]], c='r', s=50, label='End Effector')
        ax.set_title(f'{title_prefix} Frame {i}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()
        plt.pause(interval)
    
    # Use non-blocking show
    plt.show(block=False)
    plt.close(fig)  # Close the figure to free memory


def plot_all_sequences_side_by_side(ee_orig, dlo_orig, ee_norm, dlo_norm, ee_dn, dlo_dn, frame_idx=0):
    """Plot all three sequences side by side for a specific frame."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    datasets = [
        (ee_orig, dlo_orig, 'Original'),
        (ee_norm, dlo_norm, 'Normalized'), 
        (ee_dn, dlo_dn, 'Denormalized')
    ]
    
    for i, (ee_states, dlo_states, title) in enumerate(datasets):
        ax = axes[i]
        pts = dlo_states[frame_idx]
        ax.plot(pts[:,0], pts[:,1], 'o-', label='DLO')
        ee = ee_states[frame_idx][:2]
        ax.scatter([ee[0]], [ee[1]], c='r', s=50, label='End Effector')
        ax.set_title(f'{title} Frame {frame_idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_animated_comparison(ee_orig, dlo_orig, action_orig, ee_norm, dlo_norm, action_norm, ee_dn, dlo_dn, action_dn, interval=0.5):
    """Plot animated comparison of all three sequences side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    datasets = [
        (ee_orig, dlo_orig, action_orig, 'Original'),
        (ee_norm, dlo_norm, action_norm, 'Normalized'), 
        (ee_dn, dlo_dn, action_dn, 'Denormalized')
    ]
    
    # Precompute axis limits for each dataset
    limits = []
    x_min, x_max = 0.4, 0.6
    y_min, y_max = -0.1, 0.1
    limits.append((x_min, x_max, y_min, y_max))  # Placeholder for limits
    x_min, x_max = -0.1, 0.1
    y_min, y_max = -0.1, 0.1
    limits.append((x_min, x_max, y_min, y_max))  # Placeholder for limits
    x_min, x_max = 0.4, 0.6
    y_min, y_max = -0.1, 0.1
    limits.append((x_min, x_max, y_min, y_max))  # Placeholder for limits
    # for ee_states, dlo_states, title in datasets:
    #     all_pts = dlo_states.reshape(-1, 3)
    #     x_min, y_min = all_pts[:,0].min(), all_pts[:,1].min()
    #     x_max, y_max = all_pts[:,0].max(), all_pts[:,1].max()
    #     limits.append((x_min, x_max, y_min, y_max))
    
    num_frames = min(len(ee_orig), len(dlo_orig))
    
    for frame_idx in range(num_frames):
        for i, (ee_states, dlo_states, action, title) in enumerate(datasets):
            ax = axes[i]
            ax.clear()
            pts = dlo_states[frame_idx]
            ax.plot(pts[:,0], pts[:,1], 'o-', label='DLO', linewidth=2, markersize=6)
            ee = ee_states[frame_idx][:2]
            action = action[frame_idx][:2] + ee
            ax.scatter([ee[0]], [ee[1]], c='g', s=100, label='End Effector', marker='o')
            ax.scatter([action[0]], [action[1]], c='r', s=100, label='Action', marker='*')
            ax.set_title(f'{title} Frame {frame_idx}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set consistent limits
            x_min, x_max, y_min, y_max = limits[i]
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.pause(interval)
    
    plt.show()


def main():
    # Configuration
    dataset_path = "/home/lar/Riccardo/diffusion_mk2/zarr_data/test.zarr.zip"
    obs_ee_dim = 5        # [x, y, z, θ, grip]
    obs_shape_dim = 45    # 15 points * 3
    obs_target_dim = 45

    # Load and extract
    dataset, dataloader = load_dataset(dataset_path, obs_ee_dim, obs_shape_dim, obs_target_dim)
    ee_orig, dlo_orig, action_orig, ee_norm, dlo_norm, action_norm = extract_all_states(dataset_path, dataloader, obs_ee_dim, obs_shape_dim)

    # Compute normalization factors
    dlo_processor = DloDataProcessor(dlo_orig)
    ee_processor = EEStateDataProcessor(ee_orig)
    action_processor = ActionDataProcessor(action_orig, num_points=obs_shape_dim // 3, is_first_idx=False, is_last_gripper=True)
    cs0_list, csR_list = dlo_processor.compute_normalize_factors_arrays()
    dlo_processor.set_normalize_factors_arrays(cs0_list, csR_list)
    ee_processor.set_normalize_factors_arrays(cs0_list, csR_list)

    # Denormalize normalized data
    dlo_dn = []
    ee_dn = []
    action_dn = []
    for ee_n, dlo_n, action_n, cs0, csR in zip(ee_norm, dlo_norm, action_norm, cs0_list, csR_list):
        ee_dn.append(ee_processor.denormalize(ee_n, cs0, csR))
        dlo_dn.append(dlo_processor.denormalize(dlo_n, cs0, csR))
        action_dn.append(action_processor.denormalize(action_n, csR))
    dlo_dn = np.array(dlo_dn)
    ee_dn = np.array(ee_dn)
    action_dn = np.array(action_dn)

    # Print some statistics
    print(f"Number of frames: {len(ee_orig)}")
    print(f"EE state shape: {ee_orig.shape}")
    print(f"DLO state shape: {dlo_orig.shape}")
    print(f"Action shape: {action_orig.shape}")

    # Option 1: Plot individual sequences (non-blocking)
    print("Showing individual sequences...")
    # plot_sequence_xy(ee_orig, dlo_orig, 'Original', interval=0.1)
    # plot_sequence_xy(ee_norm, dlo_norm, 'Normalized', interval=0.1)
    # plot_sequence_xy(ee_dn, dlo_dn, 'Denormalized', interval=0.1)
    
    # # Option 2: Plot single frame side-by-side comparison
    # print("Showing side-by-side comparison for frame 0...")
    # plot_all_sequences_side_by_side(ee_orig, dlo_orig, ee_norm, dlo_norm, ee_dn, dlo_dn, frame_idx=0)
    
    # # Option 3: Plot animated side-by-side comparison
    # print("Showing animated side-by-side comparison...")
    plot_animated_comparison(ee_orig, dlo_orig, action_orig, ee_norm, dlo_norm, action_norm, ee_dn, dlo_dn, action_dn, interval=0.1)
    
    # Keep the script running to see all plots
    input("Press Enter to exit...")


if __name__ == '__main__':
    main()