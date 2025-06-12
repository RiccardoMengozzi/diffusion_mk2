import os
import numpy as np
import zarr

# ------------------------------------------------------------
# CONFIGURE THESE PATHS AS NEEDED
# ------------------------------------------------------------

project_dir   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
npz_filename  = os.path.join(project_dir, "npz_data/pushing_dataset.npz")
zarr_filename = os.path.join(project_dir, "zarr_data/pushing_dataset.zarr.zip")
# ------------------------------------------------------------

def create_zarr_from_npz(npz_path: str, zarr_path: str):
    """
    Reads 'observations', 'actions', and 'episode_ends' from the given .npz,
    then writes them into a compressed Zarr-Zip store with this structure:

    my_data.zarr.zip/
      ├── data/
      │    ├── state          (shape: [N, obs_dim],   dtype=float32)
      │    └── action         (shape: [N, action_dim],dtype=float32)
      └── meta/
           └── episode_ends   (shape: [E],            dtype=int64)
    """
    # 1) Load arrays from .npz
    data = np.load(npz_path)
    if "observations" not in data or "actions" not in data or "episode_ends" not in data:
        raise KeyError(
            "The .npz must contain exactly these three keys: "
            "'observations', 'actions', and 'episode_ends'."
        )

    observations = data["observations"]    # shape (N, O+1, 2)
    actions      = data["actions"]         # shape (N, 2)
    episode_ends  = data["episode_ends"]   # shape (E,)


    N_obs = observations.shape[0]
    N_act = actions.shape[0]
    if N_obs != N_act:
        raise ValueError(
            f"Length mismatch: observations has {N_obs}, actions has {N_act}"
        )

    # 2) Flatten observations into (N, obs_dim)
    #    Here obs_dim = (O+1)*2
    obs_flat = observations.reshape(N_obs, -1).astype("float32")
    act_flat = actions.astype("float32")
    ep_ends  = episode_ends.astype("int64")

    # 3) Remove any existing Zarr‐Zip so we can create afresh
    if os.path.exists(zarr_path):
        print(f"Removing existing store at {zarr_path} …")
        os.remove(zarr_path)

    # 4) Create a new ZipStore and root group
    zstore = zarr.ZipStore(zarr_path, mode="w")
    root = zarr.group(store=zstore, overwrite=True)

    # 5) Create subgroups "data" and "meta"
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")

    # 6) Determine chunk shapes (tune chunk sizes to your preference)
    obs_dim = obs_flat.shape[1]
    act_dim = act_flat.shape[1]
    chunk_samples = min(100, N_obs)

    state_chunks  = (chunk_samples, obs_dim)
    action_chunks = (chunk_samples, act_dim)

    # 7) Create the two datasets under data/
    data_grp.create_dataset(
        name="state",
        shape=(N_obs, obs_dim),
        chunks=state_chunks,
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    data_grp.create_dataset(
        name="action",
        shape=(N_obs, act_dim),
        chunks=action_chunks,
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )

    # 8) Create the episode_ends dataset under meta/
    meta_grp.create_dataset(
        name="episode_ends",
        data=ep_ends,
        dtype="int64",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )

    # 9) Write data into the Zarr datasets
    data_grp["state"][:]  = obs_flat
    data_grp["action"][:] = act_flat
    # episode_ends was passed to create_dataset, so it’s already stored

    zstore.close()
    print(f"Successfully wrote Zarr‐Zip store to: {zarr_path}")
    print(f"  data/state    shape = {obs_flat.shape}, dtype=float32")
    print(f"  data/action   shape = {act_flat.shape}, dtype=float32")
    print(f"  meta/episode_ends shape = {ep_ends.shape}, dtype=int64")


if __name__ == "__main__":
    print("==> Reading pushing_dataset.npz and writing to Zarr‐Zip …")
    create_zarr_from_npz(npz_path=npz_filename, zarr_path=zarr_filename)
