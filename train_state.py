import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gdown
import wandb

from diffusers.training_utils import EMAModel
from diffusers import get_scheduler
from diffusers.schedulers import DDPMScheduler

from diffusion_mk2.model.diffusion.conditional_unet_1d import ConditionalUnet1D
from diffusion_mk2.dataset.pusht_state_dataset import PushTStateDataset


hyperparameters = {
    "obs_dim": 32,
    "obs_horizon": 2,
    "action_dim": 2,
    "action_horizon": 8,
    "pred_horizon": 16,
    "num_diffusion_iters": 100,
    "num_epochs": 1000,
    "batch_size": 2048,
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "warmup_steps": 500,
    "ema_power": 0.75,
    "device": torch.device("cuda"),  # Will default to CUDA if available
    "model_save_path": "pushing_model.pt",
    "dataset_url_id": "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq",
    "dataset_filename": "pushing_dataset.zarr.zip",

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


        self.DEVICE = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.DATASET_FILENAME = config.get("dataset_filename", "pushing_dataset.zarr.zip")
        self.MODEL_SAVE_PATH = config.get("model_save_path", "ema_diffusion_model.pt")
        
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

        # Download dataset if needed
        self.dataset_path = self.DATASET_FILENAME
        self._ensure_dataset_downloaded()

        # Build dataset and dataloader
        self.dataset = PushTStateDataset(
            dataset_path=self.dataset_path,
            pred_horizon=self.PRED_HORIZON,
            obs_horizon=self.OBS_HORIZON,
            action_horizon=self.ACTION_HORIZON
        )
        self.stats = self.dataset.stats

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

    def _ensure_dataset_downloaded(self):
        if not os.path.isfile(self.dataset_path):
            gdown.download(
                id=self.dataset_url_id,
                output=self.dataset_path,
                quiet=False
            )

    def train(self):
        """
        Run the training loop for NUM_EPOCHS. After training completes,
        the EMA‐weighted model is copied and saved to `self.model_save_path`.
        """
        with tqdm(range(self.NUM_EPOCHS), desc="Epoch") as epoch_bar:
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

        # After all epochs: copy EMA weights and save model
        ema_model = ConditionalUnet1D(
            input_dim=self.ACTION_DIM,
            global_cond_dim=self.OBS_DIM * self.OBS_HORIZON,
        ).to(self.DEVICE)
        self.ema.copy_to(ema_model.parameters())
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
        print(f"EMA model saved to {self.MODEL_SAVE_PATH}")

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
    # Example usage; replace dataset_url_id with your actual Google Drive ID (without extra query params)
    DATASET_URL_ID = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq"
    trainer = DiffusionTrainer(config=hyperparameters)
    trainer.train()
