

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gdown
import wandb
import argparse

from diffusers.training_utils import EMAModel
from diffusers import get_scheduler
from diffusers.schedulers import DDPMScheduler

from diffusion_mk2.model.diffusion.conditional_unet_1d import ConditionalUnet1D
from diffusion_mk2.dataset.pusht_state_dataset import PushTStateDataset

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

hyperparameters = {
    "obs_dim": 62,
    "obs_horizon": 2,
    "action_dim": 2,
    "action_horizon": 8,
    "pred_horizon": 16,
    "num_diffusion_iters": 100,
    "num_epochs": 1,
    "batch_size": 1024,
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "warmup_steps": 500,
    "ema_power": 0.75,
    "device": torch.device("cuda"),  # Will default to CUDA if available
    "model_save_path": "",
    "checkpoint_save_interval": 1,  # Save checkpoint every N epochs
    "dataset_path": os.path.join(PROJECT_DIR, "zarr_data", "pushing_dataset.zarr.zip"),

    # wandb
    "project_name": "diffusion_model",
    "entity": "riccardo_mengozzi",
}

class DiffusionTrainer:
    def __init__(
        self,
        config: dict,
    ):
        
        self.OBS_DIM = config.get("obs_dim", 32)
        self.OBS_HORIZON = config.get("obs_horizon", 2)
        self.ACTION_DIM = config.get("action_dim", 2)
        self.ACTION_HORIZON = config.get("action_horizon", 8)   
        self.PRED_HORIZON = config.get("pred_horizon", 16)
        self.NUM_DIFFUSION_ITERS = config.get("num_diffusion_iters", 100)
        self.NUM_EPOCHS = config.get("num_epochs", 10)
        self.BATCH_SIZE = config.get("batch_size", 16)
        self.LR = config.get("lr", 1e-4)
        self.WEIGHT_DECAY = config.get("weight_decay", 1e-6)
        self.WARMUP_STEPS = config.get("warmup_steps", 500)
        self.EMA_POWER = config.get("ema_power", 0.75)
        self.CHECKPOINT_SAVE_INTERVAL = config.get("checkpoint_save_interval", 5)

        self.DEVICE = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.DATASET_PATH = config.get("dataset_path", "zarr_data/dummy.zarr.zip")
        
        #wandb
        self.project_name = config.get("project_name", "diffusion_model")
        self.entity = config.get("entity", "riccardo_mengozzi")

        # Initialize wandb
        self.run = wandb.init(
            config=config,
            project=self.project_name,
            entity=self.entity,
            mode="disabled"
        )
        if config.get("model_save_path") == "":
            # Default model save path if not specified
            config["model_save_path"] = f"weights/{self.run.name}_model.pt"
        self.MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, config.get("model_save_path"))
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(PROJECT_DIR, f"checkpoints/{self.run.name}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Build dataset and dataloader
        self.dataset = PushTStateDataset(
            dataset_path=self.DATASET_PATH,
            pred_horizon=self.PRED_HORIZON,
            obs_horizon=self.OBS_HORIZON,
            action_horizon=self.ACTION_HORIZON
        )
        self.stats = self.dataset.stats

        print(f"LENGTH OF DATASET: {len(self.dataset)}")

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.BATCH_SIZE,
            num_workers=1,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

        # Build diffusion components
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.NUM_DIFFUSION_ITERS,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon"
        )

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.ACTION_DIM,
            global_cond_dim=self.OBS_DIM * self.OBS_HORIZON,
        ).to(self.DEVICE)

        # EMA wrapper
        self.ema = EMAModel(
            parameters=self.noise_pred_net.parameters(),
            power=self.EMA_POWER
        )

        # Optimizer + LR scheduler
        self.optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(),
            lr=self.LR,
            weight_decay=self.WEIGHT_DECAY
        )
        total_training_steps = len(self.dataloader) * self.NUM_EPOCHS
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.WARMUP_STEPS,
            num_training_steps=total_training_steps
        )

        # Tracking
        self.global_step = 0


    def save_checkpoint(self, epoch: int, avg_loss: float):
        """
        Save a training checkpoint including model, optimizer, scheduler states
        """
        # Create EMA model for checkpoint
        ema_model = ConditionalUnet1D(
            input_dim=self.ACTION_DIM,
            global_cond_dim=self.OBS_DIM * self.OBS_HORIZON,
        ).to(self.DEVICE)
        self.ema.copy_to(ema_model.parameters())
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.noise_pred_net.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "avg_loss": avg_loss,
            "config": {
                "obs_dim": self.OBS_DIM,
                "obs_horizon": self.OBS_HORIZON,
                "action_dim": self.ACTION_DIM,
                "action_horizon": self.ACTION_HORIZON,
                "pred_horizon": self.PRED_HORIZON,
                "noise_scheduler_config": self.noise_scheduler.config,
                "dataset_stats": self.stats,
            }
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"chkp_{self.run.name}_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Log checkpoint to wandb
        wandb.log({"checkpoint_epoch": epoch, "checkpoint_avg_loss": avg_loss}, step=self.global_step)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a training checkpoint to resume training
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.DEVICE, weights_only=False)
        
        self.noise_pred_net.load_state_dict(checkpoint["model_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}, global step {self.global_step}")
        
        return checkpoint["epoch"]

    def train(self, checkpoint_path: str = None):
        """
        Run the training loop for NUM_EPOCHS. After training completes,
        the EMA‐weighted model is copied and saved to `self.model_save_path`.
        
        Args:
            resume_from_checkpoint: Path to checkpoint file to resume from
        """
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch = self.load_checkpoint(checkpoint_path) + 1
        
        with tqdm(range(start_epoch, self.NUM_EPOCHS), desc="Epoch") as epoch_bar:
            for epoch_idx in epoch_bar:
                epoch_losses = []
                with tqdm(self.dataloader, desc="Batch", leave=False) as batch_bar:
                    for batch in batch_bar:
                        loss_value = self._train_on_batch(batch)
                        epoch_losses.append(loss_value)
                        batch_bar.set_postfix(loss=loss_value)
                        wandb.log({"loss": loss_value}, step=self.global_step)
                        self.global_step += 1

                avg_loss = float(np.mean(epoch_losses))
                epoch_bar.set_postfix(avg_loss=avg_loss)
                wandb.log({"epoch_avg_loss": avg_loss}, step=self.global_step)
                
                # Save checkpoint at specified intervals
                if (epoch_idx + 1) % self.CHECKPOINT_SAVE_INTERVAL == 0:
                    self.save_checkpoint(epoch_idx, avg_loss)

        # After all epochs: save final model
        self.save_final_model()

    def save_final_model(self):
        """
        Save the final EMA model after training completion
        """
        # Create EMA model for final save
        ema_model = ConditionalUnet1D(
            input_dim=self.ACTION_DIM,
            global_cond_dim=self.OBS_DIM * self.OBS_HORIZON,
        ).to(self.DEVICE)
        self.ema.copy_to(ema_model.parameters())
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)
        
        torch.save(
            {
                "model_state_dict": ema_model.state_dict(),
                "obs_dim": self.OBS_DIM,
                "obs_horizon": self.OBS_HORIZON,
                "action_dim": self.ACTION_DIM,
                "action_horizon": self.ACTION_HORIZON,
                "pred_horizon": self.PRED_HORIZON,
                "noise_scheduler_config": self.noise_scheduler.config,
                "dataset_stats": self.stats,
            },
            self.MODEL_SAVE_PATH
        )
        print(f"Final EMA model saved to {self.MODEL_SAVE_PATH}")

    def _train_on_batch(self, batch: dict) -> float:
        """
        Performs one gradient step on a single batch and updates EMA.
        Returns the scalar loss value.
        """
        self.noise_pred_net.train()

        # Move data to device
        obs = batch["obs"].to(self.DEVICE)  # shape: (B, OBS_HORIZON, OBS_DIM) + possibly future
        actions = batch["action"].to(self.DEVICE)  # shape: (B, ACTION_HORIZON, ACTION_DIM)
        batch_size = obs.shape[0]

        # Flatten observation for FiLM conditioning
        obs_cond = obs[:, : self.OBS_HORIZON, :].flatten(start_dim=1)  # (B, OBS_HORIZON * OBS_DIM)

        # Sample random noise and timesteps
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.DEVICE
        ).long()

        # Add noise (forward process)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        # Predict noise
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

        # Compute L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # Backpropagate and step optimizer + scheduler
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        # Update EMA weights
        self.ema.step(self.noise_pred_net.parameters())

        return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model for push-t state prediction.")
    parser.add_argument("--chkp_path", type=str, default=None,
                        help="Name of the checkpoint to resume from. If not provided, training starts from scratch.")
    args = parser.parse_args()
    # Example usage; replace dataset_url_id with your actual Google Drive ID (without extra query params)
    trainer = DiffusionTrainer(config=hyperparameters)
    
    # To resume from a checkpoint, uncomment and provide the path:
    # trainer.train(resume_from_checkpoint="checkpoints/your_run_name/checkpoint_epoch_10.pt")
    
    trainer.train(checkpoint_path=args.chkp_path)
