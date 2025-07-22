import zipfile
import pickle
import numpy as np
import os
import zarr

zip_path = '/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/train3.zip'
zarr_path = '/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/train3.zarr.zip'
MAX_NUMBER_OF_EPISODES = 1e9

def compute_ee_theta(dlo_trajectory, action_idx):
    """
    Compute the end-effector theta by calculating the angle between consecutive DLO points
    and taking the orthogonal to that angle.
    
    Args:
        dlo_trajectory: DLO trajectory data of shape (time_steps, num_points, 2)
        action_idx: Index of the action point in the DLO
    
    Returns:
        ee_theta: Array of angles (orthogonal to the direction between consecutive DLO points)
    """
    ee_theta = []
    
    for t in range(dlo_trajectory.shape[0]):
        # Get the current DLO state at time t
        dlo_state = dlo_trajectory[t]  # Shape: (num_points, 2)
        
        # Calculate direction vector between dlo_state[action_idx] and dlo_state[action_idx+1]
        if action_idx + 1 < dlo_state.shape[0]:
            # Direction vector from point idx to point idx+1
            direction = dlo_state[action_idx + 1] - dlo_state[action_idx]
            
            # Calculate angle with respect to horizontal (x-axis)
            angle = np.arctan2(direction[1], direction[0])
            
            # Take orthogonal angle (add π/2 or 90 degrees)
            orthogonal_angle = angle + np.pi/2
            
            # Normalize angle to [-π, π]
            orthogonal_angle = np.arctan2(np.sin(orthogonal_angle), np.cos(orthogonal_angle))
            
        else:
            # If action_idx is the last point, use the previous direction
            if action_idx > 0:
                direction = dlo_state[action_idx] - dlo_state[action_idx - 1]
                angle = np.arctan2(direction[1], direction[0])
                orthogonal_angle = angle + np.pi/2
                orthogonal_angle = np.arctan2(np.sin(orthogonal_angle), np.cos(orthogonal_angle))
            else:
                # Default to 0 if only one point
                orthogonal_angle = 0.0
        
        ee_theta.append(orthogonal_angle)
    
    return np.array(ee_theta)

def convert(obj):
    total_action = obj["action"]
    dlo_trajectory = obj["observation"]
    target = obj["final_shape"]
    dlo_trajectory = np.moveaxis(dlo_trajectory, -1, 1) 
    target = np.moveaxis(target, -1, 0)  

    # Get the action index and delta actions
    action_idx = int(total_action[0])
    delta_actions = total_action[1:] / len(dlo_trajectory)
    delta_actions = np.insert(delta_actions, 2, 0.0) # ro simulate z
    delta_actions = np.insert(delta_actions, 4, 0.0) # to simulate gripper

    # Compute EE state (XY + theta)
    ee_xy = dlo_trajectory[:, action_idx, :2]
    ee_theta = compute_ee_theta(dlo_trajectory, action_idx)
    ee_state = np.column_stack([ee_xy, ee_theta])  # Shape: (time_steps, 3)ù
    ee_state = np.insert(ee_state, 2, 0.0, axis=1) # to simulate z
    ee_state = np.insert(ee_state, 4, 0.0, axis=1) # to simulate gripper


    # Prepare observations for each timestep
    observations = []
    actions = []
    idxs = []
    
    for t in range(dlo_trajectory.shape[0]):
        # Flatten each component for this timestep
        obs_ee = ee_state[t].flatten()  # Shape: (3,) -> EE x, y, theta
        obs_dlo = dlo_trajectory[t].flatten()  # Shape: (num_points * 2,)
        obs_target = target.flatten()  # Shape: (num_points * 2,)
        
        # Concatenate all observations for this timestep
        obs = np.concatenate([obs_ee, obs_dlo, obs_target], axis=0)
        observations.append(obs)
        actions.append(delta_actions)        


        idxs.append(action_idx)

    
    return np.array(observations), np.array(actions), np.array(idxs)

def main():
    total_observations = []
    total_actions = []
    total_idxs = []
    total_episode_ends = []
    episode_end = 0
    counter = 0

    with zipfile.ZipFile(zip_path, 'r') as z:
        # Loop through all files ending with .pkl
        for file_name in z.namelist():
            if counter > MAX_NUMBER_OF_EPISODES:
                break
            if file_name.endswith('.pkl'):
                with z.open(file_name) as f:
                    obj = pickle.load(f)
                    observations, actions, idxs = convert(obj)   
                    
                    episode_end += len(observations)     

                    total_observations.append(observations)
                    total_actions.append(actions)
                    total_idxs.append(idxs)
                    total_episode_ends.append(episode_end)
            counter += 1
                    
    # Concatenate all episodes
    observations = np.concatenate(total_observations, axis=0)
    actions = np.concatenate(total_actions, axis=0)
    idxs = np.concatenate(total_idxs, axis=0)
    episode_ends = np.array(total_episode_ends)

    print(f"Total concatenated data:")
    print(f"  Observations shape: {observations.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Idxs shape: {idxs.shape}")
    print(f"  Episode ends: {episode_ends}")


    #### CREATE ZARR ####
    N_obs = observations.shape[0]
    N_act = actions.shape[0]
    if N_obs != N_act:
        raise ValueError(
            f"Length mismatch: observations has {N_obs}, actions has {N_act}"
        )

    # Prepare data for zarr
    obs_flat = observations.astype("float32")
    act_flat = actions.astype("float32")
    idxs_flat = idxs.astype("int64")
    ep_ends = episode_ends.astype("int64")

    # Remove any existing Zarr-Zip so we can create afresh
    if os.path.exists(zarr_path):
        print(f"Removing existing store at {zarr_path} …")
        os.remove(zarr_path)

    # Create a new ZipStore and root group
    zstore = zarr.ZipStore(zarr_path, mode="w")
    root = zarr.group(store=zstore, overwrite=True)

    # Create subgroups "data" and "meta"
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")

    # Determine chunk shapes (tune chunk sizes to your preference)
    obs_dim = obs_flat.shape[1]
    act_dim = act_flat.shape[1] if len(act_flat.shape) > 1 else 1
    chunk_samples = min(100, N_obs)

    state_chunks = (chunk_samples, obs_dim)
    action_chunks = (chunk_samples, act_dim) if act_dim > 1 else (chunk_samples,)
    idx_chunks = (chunk_samples,)

    # Create the datasets under data/
    data_grp.create_dataset(
        name="state",
        shape=(N_obs, obs_dim),
        chunks=state_chunks,
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    
    if len(act_flat.shape) > 1:
        data_grp.create_dataset(
            name="action",
            shape=(N_obs, act_dim),
            chunks=action_chunks,
            dtype="float32",
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )
    else:
        data_grp.create_dataset(
            name="action",
            shape=(N_obs,),
            chunks=(chunk_samples,),
            dtype="float32",
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )

    data_grp.create_dataset(
        name="idx",
        shape=(N_obs,),
        chunks=idx_chunks,
        dtype="int64",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )

    # Create the episode_ends dataset under meta/
    meta_grp.create_dataset(
        name="episode_ends",
        data=ep_ends,
        dtype="int64",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )

    # Write data into the Zarr datasets
    data_grp["state"][:] = obs_flat
    data_grp["action"][:] = act_flat
    data_grp["idx"][:] = idxs_flat

    zstore.close()
    print(f"Successfully wrote Zarr-Zip store to: {zarr_path}")
    print(f"  data/state    shape = {obs_flat.shape}, dtype=float32")
    print(f"  data/action   shape = {act_flat.shape}, dtype=float32")
    print(f"  data/idx      shape = {idxs_flat.shape}, dtype=int64")
    print(f"  meta/episode_ends shape = {ep_ends.shape}, dtype=int64")

if __name__ == "__main__":
    print(f"==> Reading {zip_path} and writing to {zarr_path} …")
    main()