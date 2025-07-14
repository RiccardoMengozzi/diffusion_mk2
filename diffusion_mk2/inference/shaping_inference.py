"""
Simplified Diffusion-Based Action Prediction Inference Module

This module provides a streamlined class for loading and running inference with 
a pre-trained diffusion model for robotic action prediction.
"""

import torch
import numpy as np
from diffusers import DDPMScheduler
from diffusion_mk2.model.diffusion.conditional_unet_1d import ConditionalUnet1D
from diffusion_mk2.model import normalization_pca
from typing import Tuple, Dict, Union, Any


class ShapingInference:
    """
    Simplified diffusion model inference for robotic action prediction.
    
    Loads a pre-trained diffusion model and generates action trajectories
    from observation sequences using a denoising diffusion process.
    """

    def __init__(self, ckp_path: str, device: Union[torch.device, str, None] = None, verbose: bool = True):
        """
        Initialize the inference model.

        Args:
            ckp_path: Path to the model checkpoint file
            device: Target device for computation (auto-detects if None)
            verbose: Whether to print initialization progress
        """
        self.verbose = verbose
        self.cs0 = None
        self.csR = None
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        # Load checkpoint
        self._load_checkpoint(ckp_path)
    
        
        # Initialize model and scheduler
        self._initialize_model()
        self._initialize_scheduler()
        
        if self.verbose:
            self._print_model_info()

    def _load_checkpoint(self, ckp_path: str) -> None:
        """Load and extract checkpoint data."""
        if self.verbose:
            print(f"Loading checkpoint from: {ckp_path}")
        
        try:
            ckpt = torch.load(ckp_path, map_location=self.device, weights_only=False)
            
            # Extract required parameters
            self.model_state_dict = ckpt["model_state_dict"]
            self.obs_dim = ckpt["obs_dim"]
            self.obs_ee_dim = ckpt["obs_ee_dim"]
            self.obs_dlo_dim = ckpt["obs_dlo_dim"]
            self.obs_target_dim = ckpt["obs_target_dim"]
            self.obs_horizon = ckpt["obs_horizon"]
            self.action_dim = ckpt["action_dim"]
            self.action_horizon = ckpt["action_horizon"]
            self.pred_horizon = ckpt["pred_horizon"]
            self.noise_scheduler_config = ckpt["noise_scheduler_config"]
            self.dataset_stats = ckpt["dataset_stats"]
            
            if self.verbose:
                print("✓ Checkpoint loaded successfully")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def _initialize_model(self) -> None:
        """Initialize the diffusion model."""
        try:
            global_cond_dim = self.obs_dim * self.obs_horizon
            self.model = ConditionalUnet1D(
                input_dim=self.action_dim, 
                global_cond_dim=global_cond_dim
            ).to(self.device)
            
            self.model.load_state_dict(self.model_state_dict)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")

    def _initialize_scheduler(self) -> None:
        """Initialize the DDPM noise scheduler."""
        try:
            self.noise_scheduler = DDPMScheduler.from_config(self.noise_scheduler_config)
            
            # Move scheduler tensors to device
            for attr_name, attr_val in vars(self.noise_scheduler).items():
                if isinstance(attr_val, torch.Tensor):
                    setattr(self.noise_scheduler, attr_name, attr_val.to(self.device))
                    
        except Exception as e:
            raise RuntimeError(f"Failed to initialize scheduler: {e}")

    def _print_model_info(self) -> None:
        """Print model configuration."""
        print(f"\n{'='*50}")
        print("MODEL CONFIGURATION")
        print(f"{'='*50}")
        print(f"Observation dimension: {self.obs_dim}")
        print(f"Observation horizon: {self.obs_horizon}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Action horizon: {self.action_horizon}")
        print(f"Prediction horizon: {self.pred_horizon}")
        print(f"Global condition dim: {self.obs_dim * self.obs_horizon}")
        print(f"Device: {self.device}")
        print(f"{'='*50}\n")

    def _prepare_observation(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Prepare and validate observation input."""
        # Convert to tensor and move to device
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        elif isinstance(observation, torch.Tensor):
            observation = observation.to(self.device, dtype=torch.float32)
        else:
            raise TypeError("Observation must be a numpy array or torch tensor")

        # Validate shape
        expected_shape = (self.obs_horizon, self.obs_dim)
        if observation.shape != expected_shape:
            raise ValueError(f"Expected observation shape {expected_shape}, got {observation.shape}")

        return observation

    def _process_observation_conditioning(self, observation: torch.Tensor) -> torch.Tensor:
        """Process observation for model conditioning."""
        # Extract components
        ee_states = observation[:, :self.obs_ee_dim]
        ee_states_pos = ee_states[:, :3]  # [x, y, z]
        ee_states_theta = ee_states[:, 3]  # [theta]
        ee_states_gripper = ee_states[:, 4]  # [gripper]
        
        initial_shape_start = self.obs_ee_dim
        initial_shape_end = initial_shape_start + self.obs_dlo_dim
        initial_shapes = observation[:, initial_shape_start:initial_shape_end].reshape(
            self.obs_horizon, self.obs_dlo_dim // 3, 3
        )
        
        target_shape_start = initial_shape_end
        target_shape_end = target_shape_start + self.obs_target_dim
        target_shapes = observation[:, target_shape_start:target_shape_end].reshape(
            self.obs_horizon, self.obs_target_dim // 3, 3
        )

        
        obs_cond = []
        for ee_state_pos, ee_state_theta, ee_state_gripper, init_shape, target_shape in zip(ee_states_pos, ee_states_theta, ee_states_gripper, initial_shapes, target_shapes):
            ee_state_pos = ee_state_pos.cpu().numpy() if isinstance(ee_state_pos, torch.Tensor) else ee_state_pos
            ee_state_theta = ee_state_theta.cpu().numpy() if isinstance(ee_state_theta, torch.Tensor) else ee_state_theta
            ee_state_gripper = ee_state_gripper.cpu().numpy() if isinstance(ee_state_gripper, torch.Tensor) else ee_state_gripper
            init_shape = init_shape.cpu().numpy() if isinstance(init_shape, torch.Tensor) else init_shape
            target_shape = target_shape.cpu().numpy() if isinstance(target_shape, torch.Tensor) else target_shape

            cs0, csR = normalization_pca.compute_normalize_factors(init_shape)
            self.cs0, self.csR = cs0, csR
            
            init_shape_n = normalization_pca.normalize_pca(init_shape, cs0, csR)
            target_shape_n = normalization_pca.normalize_pca(target_shape, cs0, csR)

            ee_state_pos_n = normalization_pca.normalize_pca(ee_state_pos, cs0, csR)
            ee_state_theta_n = normalization_pca.normalize_min_max(ee_state_theta, self.dataset_stats["action"]["min"][3], self.dataset_stats["action"]["max"][3])
            ee_state_gripper_n = normalization_pca.normalize_min_max(ee_state_gripper, self.dataset_stats["action"]["min"][4], self.dataset_stats["action"]["max"][4])
            ee_state_n = np.array([ee_state_pos_n[0], ee_state_pos_n[1], ee_state_pos_n[2], ee_state_theta_n, ee_state_gripper_n])

            obs_cond.append(np.concatenate([
                ee_state_n,
                init_shape_n.flatten(),
                target_shape_n.flatten()
            ]))

        
        obs_cond = np.array(obs_cond, dtype=np.float32)

        obs_cond = obs_cond.flatten()
        return torch.tensor(obs_cond, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _denoise_actions(self, obs_cond: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Run the denoising diffusion process."""
        # Initialize with random noise
        noisy_action = torch.randn(
            (1, self.pred_horizon, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        
        # Set up denoising schedule
        num_steps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_steps)
        
        if self.verbose:
            print(f"Running {num_steps} denoising steps...")
        
        # Store intermediate results
        trajectory = []
        
        # Iterative denoising
        for i, timestep in enumerate(self.noise_scheduler.timesteps):
            # Predict noise
            noise_pred = self.model(
                sample=noisy_action, 
                timestep=timestep, 
                global_cond=obs_cond
            )
            
            # Remove predicted noise
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, 
                timestep=timestep, 
                sample=noisy_action
            ).prev_sample
            
            # Store intermediate result
            trajectory.append(noisy_action.squeeze(0).detach().cpu().numpy())
            
            # Progress indicator
            if self.verbose and (i + 1) % (num_steps // 4) == 0:
                print(f"  Completed {i + 1}/{num_steps} steps")
        
        return noisy_action, trajectory

    def _postprocess_actions(self, final_action: torch.Tensor, trajectory: list) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process the denoised predictions."""
        # Convert final prediction to numpy
        final_action = final_action.detach().cpu().numpy()[0]
        # for key, data in self.dataset_stats.items():
        #     print(f"Dataset stats for {key}: min={data['min']}, max={data['max']}")

        # Denormalize final action
        final_action_pos = final_action[:, :3]  # Assuming first 2 columns are positions
        final_action_theta = final_action[:, 3]
        final_action_gripper = final_action[:, 4]

        final_action_pos_dn = normalization_pca.denormalize_pca_batch(
            final_action_pos, self.cs0, self.csR, rotation_only=True
        )
        final_action_theta_dn = normalization_pca.denormalize_min_max_batch(
            final_action_theta, 
            self.dataset_stats["action"]["min"][3], 
            self.dataset_stats["action"]["max"][3]
        )
        final_action_gripper_dn = normalization_pca.denormalize_min_max_batch(
            final_action_gripper, 
            self.dataset_stats["action"]["min"][4], 
            self.dataset_stats["action"]["max"][4]
        )
        final_action_dn = np.column_stack([
            final_action_pos_dn[:, 0],
            final_action_pos_dn[:, 1],
            final_action_pos_dn[:, 2],
            final_action_theta_dn,
            final_action_gripper_dn
        ])


        # Extract executable actions (receding horizon)
        start_idx = self.obs_horizon - 1
        end_idx = start_idx + self.action_horizon
        executable_actions = final_action_dn[start_idx:end_idx, :]
        
        # Process full trajectory
        full_trajectory = []
        for actions in trajectory:
            actions_pos = actions[:, :3]  # Assuming first 2 columns are positions
            actions_theta = actions[:, 3]
            actions_gripper = actions[:, 4]

            actions_pos_dn = normalization_pca.denormalize_pca_batch(
                actions_pos, self.cs0, self.csR, rotation_only=True
            )
            actions_theta_dn = normalization_pca.denormalize_min_max_batch(
                actions_theta, 
                self.dataset_stats["action"]["min"][3], 
                self.dataset_stats["action"]["max"][3]
            )
            actions_gripper_dn = normalization_pca.denormalize_min_max_batch(
                actions_gripper, 
                self.dataset_stats["action"]["min"][4], 
                self.dataset_stats["action"]["max"][4]
            )
            actions_dn = np.column_stack([
                actions_pos_dn[:, 0],
                actions_pos_dn[:, 1],
                actions_pos_dn[:, 2],
                actions_theta_dn,
                actions_gripper_dn
            ])

            full_trajectory.append(np.array(actions_dn))
        full_trajectory = np.array(full_trajectory, dtype=np.float32)

        if self.verbose:
            print(f"✓ Generated {executable_actions.shape[0]} executable actions")
            print(f"✓ Full trajectory shape: {full_trajectory.shape}")
        
        return executable_actions, full_trajectory

    def run_inference(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference to predict action sequence from observation history.

        Args:
            observation: Observation sequence of shape (obs_horizon, obs_dim)

        Returns:
            Tuple containing:
            - executable_actions: Actions for next steps (action_horizon, action_dim)
            - full_trajectory: Complete prediction trajectory (num_steps, pred_horizon, action_dim)
        """
        # Prepare input
        observation = self._prepare_observation(observation)
        
        with torch.no_grad():
            # Process observation conditioning
            obs_cond = self._process_observation_conditioning(observation)
            
            # Run denoising process
            final_action, trajectory = self._denoise_actions(obs_cond)
            
            # Post-process results
            executable_actions, full_trajectory = self._postprocess_actions(final_action, trajectory)
        
        return executable_actions, full_trajectory


# Example usage
if __name__ == "__main__":
    """Example usage of the ShapingInference class."""
    
    checkpoint_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/weights/chkp_dummy-529qk1bd_epoch_30.pt"
    

    # Initialize model
    model = ShapingInference(checkpoint_path, verbose=True)
    print("Model loaded successfully!")
    
    # Create dummy observation data
    dummy_obs = np.random.randn(model.obs_horizon, model.obs_dim)
    print(f"Input observation shape: {dummy_obs.shape}")
    
    # Run inference
    actions, trajectory = model.run_inference(dummy_obs)
    
    print(f"Executable actions shape: {actions.shape}")
    print(f"Full trajectory shape: {trajectory.shape}")
    print(f"First action to execute: {actions[0]}")
    
    # Example of silent operation
    print("\n--- Silent mode example ---")
    silent_model = ShapingInference(checkpoint_path, verbose=False)
    silent_actions, _ = silent_model.run_inference(dummy_obs)
    print("Silent inference completed successfully!")
    
