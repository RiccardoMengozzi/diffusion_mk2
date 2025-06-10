import torch
import numpy as np
from diffusers import DDPMScheduler
from diffusion_mk2.model.diffusion.conditional_unet_1d import ConditionalUnet1D


class InferenceState:
    def __init__(self, ckp_path, device=None):
        # ----------------------------
        # Configuration / file paths
        # ----------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = ckp_path  

        # ----------------------------
        # 1. Load checkpoint
        # ----------------------------
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model_state_dict       = ckpt["model_state_dict"]
        self.obs_dim                = ckpt["obs_dim"]
        self.obs_horizon            = ckpt["obs_horizon"]
        self.action_dim             = ckpt["action_dim"]
        self.action_horizon         = ckpt["action_horizon"]
        self.pred_horizon           = ckpt["pred_horizon"]
        self.noise_scheduler_config = ckpt["noise_scheduler_config"]
        self.dataset_stats          = ckpt["dataset_stats"]

        # ----------------------------
        # 2. Reconstruct model
        # ----------------------------
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon
        ).to(device)
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()

        # ----------------------------
        # 3. Reconstruct scheduler
        # ----------------------------
        self.noise_scheduler = DDPMScheduler.from_config(self.noise_scheduler_config)

        # Move all internal torch.Tensor buffers of scheduler onto `device`
        for attr_name, attr_val in vars(self.noise_scheduler).items():
            if isinstance(attr_val, torch.Tensor):
                setattr(self.noise_scheduler, attr_name, attr_val.to(device))

    
    def unnormalize_data(self, data, stats):
        """
        Unnormalize data from [-1, 1] to original range.
        :param data: (B, T, D) tensor
        :param stats: dict with 'min' and 'max' keys
        :return: unnormalized data
        """
        # Convert to [0, 1]

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        ndata = (data + 1) / 2
        # Scale to original range
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data

    
    def normalize_data(self, data, stats):
        """
        Normalize data to [-1, 1] range.
        :param data: (B, T, D) tensor
        :param stats: dict with 'min' and 'max' keys
        :return: normalized data
        """
        # Scale to [0, 1]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # Convert to [-1, 1]
        ndata = ndata * 2 - 1
        return torch.from_numpy(ndata).to(self.device, dtype=torch.float32)


    def run_inference(self, observation) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            # Convert numpy array to torch tensor
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        elif isinstance(observation, torch.Tensor):
            # Ensure tensor is on the correct device
            observation = observation.to(self.model.device)
        else:
            raise TypeError("Observation must be a numpy array or a torch tensor.")
        num_diffusion_iters = self.noise_scheduler.config.num_train_timesteps
        # Flatten to (1, OBS_HORIZON * OBS_DIM) for FiLM conditioning
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            observation = observation.reshape(self.obs_horizon, self.obs_dim)
            obs_cond = self.normalize_data(observation, stats=self.dataset_stats['obs'])
            obs_cond = obs_cond.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (1, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            nactions = []
            self.noise_scheduler.set_timesteps(num_diffusion_iters)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.model(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
                nactions.append(naction.squeeze(0).detach().to('cpu').numpy())

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
    
        naction = naction[0]
        action_pred = self.unnormalize_data(naction, stats=self.dataset_stats['action'])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # ----------------------------
        # 6. Inspect results
        # ----------------------------
        nactions = np.array(nactions)
        nactions = self.unnormalize_data(nactions, stats=self.dataset_stats['action'])
        return action, nactions
