import os
import numpy as np
import zarr
import json

# ------------------------------------------------------------
# CONFIGURE THESE PATHS AS NEEDED
# ------------------------------------------------------------

project_dir   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
filename  = os.path.join(project_dir, "json_data", "fixed_simple_fc_combined.jsonl")
zarr_filename = os.path.join(project_dir, "zarr_data/shape_prediction.zarr.zip")

# ------------------------------------------------------------

def create_zarr_from_jsonl(npz_path: str, zarr_path: str):

    # 1) Load arrays from .npz or .jsonl
    initial_shapes = []
    final_shapes = []
    actions      = []

    #### JSONL ####
    if filename.endswith(".jsonl"):
        with open(filename, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    if data.get("type") == "data":
                        # Extract observation and action
                        initial_shape = np.array(data["initial_shape"])
                        final_shape = np.array(data["final_shape"])
                        action = np.array(data["action"])
                        

                        initial_shape = initial_shape.flatten()
                        final_shape = final_shape.flatten()
                        action = action.flatten()

                        initial_shapes.append(initial_shape)
                        final_shapes.append(final_shape)
                        actions.append(action)
                    
                    
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue

        if not initial_shapes:
            raise ValueError("No valid observation/action pairs found in JSON file")

        initial_shapes = np.array(initial_shapes, dtype=np.float32)
        final_shapes = np.array(final_shapes, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)

    #### CREATE ZARR ####
    N_obs = initial_shapes.shape[0]
    N_act = actions.shape[0]
    if N_obs != N_act:
        raise ValueError(
            f"Length mismatch: observations has {N_obs}, actions has {N_act}"
        )

    # 2) Flatten observations into (N, obs_dim)
    #    Here obs_dim = (O+1)*2
    initial_shapes = initial_shapes.reshape(N_obs, -1).astype("float32")
    final_shapes = final_shapes.reshape(N_obs, -1).astype("float32")
    actions  = actions.astype("float32")

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
    shape_dim = initial_shapes.shape[1]
    action_dim = actions.shape[1]
    chunk_samples = min(100, N_obs)

    initial_shape_chunks  = (chunk_samples, shape_dim)
    final_shape_chunks    = (chunk_samples, shape_dim)
    action_chunks         = (chunk_samples, action_dim)

    # 7) Create the two datasets under data/
    data_grp.create_dataset(
        name="initial_shape",
        shape=(N_obs, shape_dim),
        chunks=initial_shape_chunks,
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    data_grp.create_dataset(
        name="final_shape",
        shape=(N_obs, shape_dim),
        chunks=final_shape_chunks,
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )

    data_grp.create_dataset(
        name="action",
        shape=(N_obs, action_dim),
        chunks=action_chunks,
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )


    # 9) Write data into the Zarr datasets
    data_grp["initial_shape"][:] = initial_shapes
    data_grp["final_shape"][:] = final_shapes
    data_grp["action"][:] = actions

    zstore.close()
    print(f"Successfully wrote Zarr‐Zip store to: {zarr_path}")
    print(f"  - Initial shapes: {initial_shapes.shape}")
    print(f"  - Final shapes: {final_shapes.shape}")
    print(f"  - Actions: {actions.shape}")


if __name__ == "__main__":
    print("==> Reading pushing_dataset.npz and writing to Zarr‐Zip …")
    create_zarr_from_jsonl(npz_path=filename, zarr_path=zarr_filename)
