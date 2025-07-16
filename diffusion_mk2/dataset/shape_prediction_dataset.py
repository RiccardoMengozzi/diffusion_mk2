import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import zarr
from diffusion_mk2.model import normalization_pca



class DloDataset(Dataset):
    def __init__(self, dataset_path, num_points=15):
        self.num_points = num_points

        dataset_root = zarr.open(dataset_path, 'r')



        train_data = {
            "actions": dataset_root["data"]["action"][:],
            "initial_shapes": dataset_root["data"]["initial_shape"][:].reshape(-1, num_points, 3),
            "final_shapes": dataset_root['data']['final_shape'][:].reshape(-1, num_points, 3)

        }

        self.stats = self.get_data_stats(train_data)
        self.normalized_train_data = self.normalize_data(train_data)


    def get_data_stats(self, data):
        dlo_states = data["initial_shapes"][:]
        final_shapes = data["final_shapes"][:]
        actions = data["actions"]


        dlo_states_stats = normalization_pca.get_data_stats(dlo_states.reshape(-1, dlo_states.shape[-1]))
        final_shapes_stats = normalization_pca.get_data_stats(final_shapes.reshape(-1, final_shapes.shape[-1]))
        actions_stats = normalization_pca.get_data_stats(actions)

        return {
            "initial_shapes": dlo_states_stats,
            "final_shapes": final_shapes_stats,
            "action": actions_stats,
        }


    def normalize_data(self, train_data):

        actions = train_data["actions"]
        initial_shapes = train_data["initial_shapes"]
        final_shapes = train_data["final_shapes"]

        actions_idx = actions[:, 0]
        actions_pos = actions[:, 1:3]  # Assuming the first 3 columns are the action positions
        actions_theta = actions[:, 3]

        normalized_samples = []
        for init_shape, final_shape, action_idx, action_pos, action_theta in tqdm(
            zip(initial_shapes, final_shapes, actions_idx, actions_pos, actions_theta),
            desc="Normalizing data", 
            total=len(initial_shapes)
        ):
            init_shape = init_shape[:, :2]
            final_shape = final_shape[:, :2]
            

            cs0, csR = normalization_pca.compute_normalize_factors(init_shape)
            init_shape_n = normalization_pca.normalize_pca(init_shape, cs0, csR)
            final_shape_n = normalization_pca.normalize_pca(final_shape, cs0, csR)
            action_idx_n = normalization_pca.normalize_min_max(action_idx, self.stats["action"]["min"][0], self.stats["action"]["max"][0])
            action_pos_n = normalization_pca.normalize_pca(action_pos, cs0, csR, rotation_only=True)
            action_theta_n = normalization_pca.normalize_min_max(action_theta, self.stats["action"]["min"][-1], self.stats["action"]["max"][-1])

            action_n = np.array([action_idx_n, action_pos_n[0], action_pos_n[1], action_theta_n])
            normalized_samples.append({
                "initial_shape": np.array(init_shape_n),
                "final_shape": np.array(final_shape_n),
                "action": np.array(action_n)
            })

        return normalized_samples


    def __len__(self):
        return len(self.normalized_train_data)

    def __getitem__(self, idx):
        return self.normalized_train_data[idx]


if __name__ == "__main__":
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/shape_prediction.zarr.zip"
    dataset = DloDataset(dataset_path, num_points=15)
    print("Dataset length:", len(dataset))
