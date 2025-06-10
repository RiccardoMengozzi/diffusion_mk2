"""
Inference State Module for Diffusion-Based Action Prediction

This module provides a class for loading and running inference with a pre-trained
diffusion model for robotic action prediction. The model takes observation sequences
and generates action trajectories using a denoising diffusion process.

Key Components:
- ConditionalUnet1D: The neural network architecture for diffusion
- DDPMScheduler: Denoising Diffusion Probabilistic Models scheduler
- Data normalization/unnormalization utilities
- Inference pipeline for action prediction

Author: [Your Name]
Date: [Date]
"""

import torch
import numpy as np
from diffusers import DDPMScheduler
from diffusion_mk2.model.diffusion.conditional_unet_1d import ConditionalUnet1D
from typing import Tuple, Dict, Union, Any


class InferenceState:
    """
    A class for performing inference with a pre-trained diffusion model for action prediction.
    
    This class loads a trained diffusion model and provides methods to:
    1. Load model checkpoints with all necessary configurations
    2. Normalize and unnormalize data for model compatibility
    3. Run inference to predict action sequences from observations
    
    The model uses a conditional U-Net architecture with DDPM scheduling for
    generating smooth, realistic action trajectories conditioned on observation history.
    
    Attributes:
        device (torch.device): Computing device (CPU/GPU)
        model (ConditionalUnet1D): The loaded diffusion model
        noise_scheduler (DDPMScheduler): DDPM scheduler for denoising process
        obs_dim (int): Dimensionality of observations
        obs_horizon (int): Number of observation timesteps used for conditioning
        action_dim (int): Dimensionality of actions
        action_horizon (int): Number of action timesteps to execute
        pred_horizon (int): Number of action timesteps to predict
        dataset_stats (Dict): Statistics for data normalization (min/max values)
    """
    
    def __init__(self, ckp_path: str, device: Union[torch.device, str, None] = None, verbose: bool = True) -> None:
        """
        Initialize the InferenceState with a pre-trained model checkpoint.
        
        Args:
            ckp_path (str): Path to the model checkpoint file (.pt)
            device (Union[torch.device, str, None]): Target device for computation.
                If None, automatically selects CUDA if available, otherwise CPU.
            verbose (bool): If True, print initialization progress and model info.
                If False, run silently. Default is True.
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint is missing required keys
            RuntimeError: If model loading fails
        """
        self.verbose = verbose
        # Set computing device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        checkpoint_path = ckp_path

        # Load checkpoint and extract configuration
        if self.verbose:
            print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self._extract_checkpoint_data(ckpt)
            if self.verbose:
                print("✓ Checkpoint loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Initialize and load the diffusion model
        self._initialize_model()
        
        # Initialize the noise scheduler
        self._initialize_scheduler()
        
        if self.verbose:
            print(f"✓ Model initialized on device: {self.device}")
            self._print_model_info()

    def _extract_checkpoint_data(self, ckpt: Dict[str, Any]) -> None:
        """Extract necessary data from checkpoint dictionary."""
        required_keys = [
            "model_state_dict", "obs_dim", "obs_horizon", "action_dim", 
            "action_horizon", "pred_horizon", "noise_scheduler_config", "dataset_stats"
        ]
        
        for key in required_keys:
            if key not in ckpt:
                raise KeyError(f"Missing required key in checkpoint: {key}")
        
        self.model_state_dict = ckpt["model_state_dict"]
        self.obs_dim = ckpt["obs_dim"]
        self.obs_horizon = ckpt["obs_horizon"]
        self.action_dim = ckpt["action_dim"]
        self.action_horizon = ckpt["action_horizon"]
        self.pred_horizon = ckpt["pred_horizon"]
        self.noise_scheduler_config = ckpt["noise_scheduler_config"]
        self.dataset_stats = ckpt["dataset_stats"]

    def _initialize_model(self) -> None:
        """Initialize the ConditionalUnet1D model and load weights."""
        try:
            # Create model with conditional input dimension
            global_cond_dim = self.obs_dim * self.obs_horizon
            self.model = ConditionalUnet1D(
                input_dim=self.action_dim,
                global_cond_dim=global_cond_dim
            ).to(self.device)
            
            # Load pre-trained weights
            self.model.load_state_dict(self.model_state_dict)
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")

    def _initialize_scheduler(self) -> None:
        """Initialize the DDPM noise scheduler."""
        try:
            self.noise_scheduler = DDPMScheduler.from_config(self.noise_scheduler_config)
            
            # Move scheduler's tensor attributes to the target device
            for attr_name, attr_val in vars(self.noise_scheduler).items():
                if isinstance(attr_val, torch.Tensor):
                    setattr(self.noise_scheduler, attr_name, attr_val.to(self.device))
                    
        except Exception as e:
            raise RuntimeError(f"Failed to initialize scheduler: {e}")

    def _print_model_info(self) -> None:
        """Print model configuration information."""
        if not self.verbose:
            return
            
        print("\n" + "="*50)
        print("MODEL CONFIGURATION")
        print("="*50)
        print(f"Observation dimension: {self.obs_dim}")
        print(f"Observation horizon: {self.obs_horizon}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Action horizon: {self.action_horizon}")
        print(f"Prediction horizon: {self.pred_horizon}")
        print(f"Global condition dim: {self.obs_dim * self.obs_horizon}")
        print("="*50 + "\n")
    
    def unnormalize_data(self, data: Union[torch.Tensor, np.ndarray], 
                        stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Unnormalize data from [-1, 1] range back to original scale.
        
        The model operates on normalized data in the range [-1, 1]. This method
        converts the model output back to the original data scale using the
        min/max statistics from the training dataset.
        
        Args:
            data (Union[torch.Tensor, np.ndarray]): Normalized data in range [-1, 1]
                Shape can be (B, T, D) for batched sequences or (T, D) for single sequence
            stats (Dict[str, np.ndarray]): Dictionary containing 'min' and 'max' arrays
                for denormalization. These should match the last dimension of data.
        
        Returns:
            np.ndarray: Unnormalized data in original scale
            
        Example:
            >>> normalized_actions = model_output  # Shape: (10, 3), range [-1, 1]
            >>> original_actions = unnormalize_data(normalized_actions, action_stats)
            >>> # original_actions now in real-world units (e.g., meters, radians)
        """
        # Convert tensor to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # Convert from [-1, 1] to [0, 1]
        normalized_data = (data + 1) / 2
        
        # Scale back to original range: [0, 1] -> [min, max]
        original_data = normalized_data * (stats['max'] - stats['min']) + stats['min']
        
        return original_data

    def normalize_data(self, data: Union[torch.Tensor, np.ndarray], 
                      stats: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Normalize data to [-1, 1] range for model input.
        
        The diffusion model expects input data to be normalized to the range [-1, 1].
        This method converts real-world data to this normalized range using
        training dataset statistics.
        
        Args:
            data (Union[torch.Tensor, np.ndarray]): Raw data in original scale
                Shape can be (B, T, D) for batched sequences or (T, D) for single sequence
            stats (Dict[str, np.ndarray]): Dictionary containing 'min' and 'max' arrays
                for normalization. These should match the last dimension of data.
        
        Returns:
            torch.Tensor: Normalized data in range [-1, 1], moved to model device
            
        Example:
            >>> raw_observations = np.array([[0.5, 0.2, 0.8], [0.6, 0.3, 0.7]])
            >>> norm_obs = normalize_data(raw_observations, obs_stats)
            >>> # norm_obs is now in [-1, 1] range and on the correct device
        """
        # Convert tensor to numpy for consistent processing
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # Scale to [0, 1]: (data - min) / (max - min)
        normalized_data = (data - stats['min']) / (stats['max'] - stats['min'])
        
        # Convert to [-1, 1]: [0, 1] -> [-1, 1]
        normalized_data = normalized_data * 2 - 1
        
        # Convert back to tensor and move to device
        return torch.from_numpy(normalized_data).to(self.device, dtype=torch.float32)

    def run_inference(self, observation: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference to predict action sequence from observation history.
        
        This method implements the full diffusion inference pipeline:
        1. Normalize input observations
        2. Initialize random noise for action prediction
        3. Iteratively denoise using the trained model
        4. Unnormalize the final predictions
        5. Extract executable action horizon
        
        Args:
            observation (Union[np.ndarray, torch.Tensor]): Observation sequence
                Shape: (obs_horizon, obs_dim) - sequence of observations
                Examples: robot joint positions, end-effector poses, sensor readings
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - action (np.ndarray): Executable actions for next steps
                    Shape: (action_horizon, action_dim)
                - nactions (np.ndarray): Full prediction trajectory (all denoising steps)
                    Shape: (num_diffusion_steps, pred_horizon, action_dim)
        
        Raises:
            TypeError: If observation is not numpy array or torch tensor
            ValueError: If observation shape doesn't match expected dimensions
            
        Example:
            >>> obs_history = np.array([...])  # Shape: (obs_horizon, obs_dim)
            >>> executable_actions, full_trajectory = model.run_inference(obs_history)
            >>> # Use executable_actions for robot control
            >>> robot.execute_actions(executable_actions)
        """
        # Input validation and device placement
        observation = self._prepare_observation_input(observation)
        
        # Get number of diffusion denoising steps
        num_diffusion_iters = self.noise_scheduler.config.num_train_timesteps
        
        with torch.no_grad():
            # Prepare observation conditioning
            obs_cond = self._prepare_observation_conditioning(observation)
            
            # Initialize action prediction with Gaussian noise
            noisy_action = torch.randn(
                (1, self.pred_horizon, self.action_dim), 
                device=self.device,
                dtype=torch.float32
            )
            
            # Run iterative denoising process
            denoised_actions = self._run_denoising_loop(
                noisy_action, obs_cond, num_diffusion_iters
            )
            
        # Post-process results
        executable_actions, full_trajectory = self._postprocess_predictions(denoised_actions)
        
        return executable_actions, full_trajectory

    def _prepare_observation_input(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Validate and prepare observation input."""
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        elif isinstance(observation, torch.Tensor):
            observation = observation.to(self.device, dtype=torch.float32)
        else:
            raise TypeError("Observation must be a numpy array or torch tensor.")
        
        # Validate shape
        expected_shape = (self.obs_horizon, self.obs_dim)
        if observation.shape != expected_shape:
            raise ValueError(f"Expected observation shape {expected_shape}, got {observation.shape}")
            
        return observation

    def _prepare_observation_conditioning(self, observation: torch.Tensor) -> torch.Tensor:
        """Prepare observation for model conditioning."""
        # Normalize observations using dataset statistics
        obs_cond = self.normalize_data(observation, stats=self.dataset_stats['obs'])
        
        # Reshape for FiLM conditioning: (obs_horizon, obs_dim) -> (1, obs_horizon * obs_dim)
        obs_cond = obs_cond.unsqueeze(0).flatten(start_dim=1)
        
        return obs_cond

    def _run_denoising_loop(self, noisy_action: torch.Tensor, obs_cond: torch.Tensor, 
                           num_diffusion_iters: int) -> Tuple[torch.Tensor, list]:
        """Run the iterative denoising process."""
        naction = noisy_action
        nactions = []  # Store intermediate results for visualization
        
        # Set up the denoising schedule
        self.noise_scheduler.set_timesteps(num_diffusion_iters)
        
        if self.verbose:
            print(f"Running {num_diffusion_iters} denoising steps...")
        
        # Iterative denoising
        for i, timestep in enumerate(self.noise_scheduler.timesteps):
            # Predict noise at current timestep
            noise_pred = self.model(
                sample=naction,
                timestep=timestep,
                global_cond=obs_cond
            )
            
            # Remove predicted noise (one denoising step)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=naction
            ).prev_sample
            
            # Store intermediate result
            nactions.append(naction.squeeze(0).detach().cpu().numpy())
            
            # Progress indicator for long inference
            if self.verbose and (i + 1) % (num_diffusion_iters // 4) == 0:
                print(f"  Completed {i + 1}/{num_diffusion_iters} steps")
        
        return naction, nactions

    def _postprocess_predictions(self, denoised_actions: Tuple[torch.Tensor, list]) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process the denoised predictions."""
        final_action, all_actions = denoised_actions
        
        # Convert final prediction to numpy and unnormalize
        final_action_np = final_action.detach().cpu().numpy()[0]  # Remove batch dimension
        action_pred = self.unnormalize_data(final_action_np, stats=self.dataset_stats['action'])
        
        # Extract executable actions (typically a subset of the full prediction)
        # This implements a receding horizon approach
        start_idx = self.obs_horizon - 1  # Account for observation-action alignment
        end_idx = start_idx + self.action_horizon
        executable_actions = action_pred[start_idx:end_idx, :]
        
        # Process full trajectory for analysis/visualization
        all_actions_np = np.array(all_actions)  # Shape: (num_steps, pred_horizon, action_dim)
        full_trajectory = self.unnormalize_data(all_actions_np, stats=self.dataset_stats['action'])
        
        if self.verbose:
            print(f"✓ Generated {executable_actions.shape[0]} executable actions")
            print(f"✓ Full trajectory shape: {full_trajectory.shape}")
        
        return executable_actions, full_trajectory


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the InferenceState class.
    
    This demonstrates how to:
    1. Load a trained model
    2. Prepare observation data
    3. Run inference
    4. Use the results
    """
    
    # Initialize the inference state
    try:
        model = InferenceState("/home/mengo/Research/LLM_DOM/diffusion_mk2/circle_model.pt", verbose=True)
        print("Model loaded successfully!")
        
        # Create dummy observation data (replace with real sensor data)
        dummy_obs = np.random.randn(model.obs_horizon, model.obs_dim)
        print(f"Input observation shape: {dummy_obs.shape}")
        
        # Run inference
        actions, trajectory = model.run_inference(dummy_obs)
        
        print(f"Executable actions shape: {actions.shape}")
        print(f"Full trajectory shape: {trajectory.shape}")
        
        # Use the actions (example)
        print("First action to execute:", actions[0])
        
        # Example of silent operation
        print("\n--- Silent mode example ---")
        silent_model = InferenceState("/home/mengo/Research/LLM_DOM/diffusion_mk2/circle_model.pt", verbose=False)
        silent_actions, _ = silent_model.run_inference(dummy_obs)
        
    except Exception as e:
        print(f"Error during inference: {e}")