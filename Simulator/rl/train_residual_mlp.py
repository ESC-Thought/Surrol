import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import pytorch_lightning as pl
from datetime import datetime
import wandb
from typing import List, Dict, Optional

class CorrectionDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: Path to directory containing h5 files
            split: Either "train" or "val"
        """
        self.data_dir = data_dir
        self.split = split
        self.episode_paths = []
        
        # Collect all h5 files
        for episode_dir in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, episode_dir)):
                h5_path = os.path.join(data_dir, episode_dir, 'data.h5')
                if os.path.exists(h5_path):
                    self.episode_paths.append(h5_path)
        
        print(f"Found {len(self.episode_paths)} episodes for {split}")
        
        # Split train/val
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(self.episode_paths)
        split_idx = int(len(self.episode_paths) * 0.9)
        self.episode_paths = self.episode_paths[:split_idx] if split == "train" else self.episode_paths[split_idx:]
        
        # Load all data
        self.data = {
            "is_human_intervention": [],
            "policy_action": [],
            "policy_obs": [],
            "correction_action": [],
        }
        
        print(f"Loading {split} data...")
        for path in self.episode_paths:
            with h5py.File(path, 'r') as f:
                for key in self.data.keys():
                    self.data[key].extend(f[key][()])
                    
        # Convert to numpy arrays
        for key in self.data.keys():
            self.data[key] = np.array(self.data[key])
            
        print(f"Loaded {len(self.data['policy_obs'])} samples for {split}")
        print(f"Number of interventions: {np.sum(self.data['is_human_intervention'])}")

    def __len__(self):
        return len(self.data["policy_obs"])

    def __getitem__(self, idx):
        return {
            "is_intervention": torch.FloatTensor([self.data["is_human_intervention"][idx]]),
            "policy_action": torch.FloatTensor(self.data["policy_action"][idx]),
            "obs": torch.FloatTensor(self.data["policy_obs"][idx]),
            "correction": torch.FloatTensor(self.data["correction_action"][idx])
        }

class MLPResidualPolicy(nn.Module):
    """Simple MLP-based residual policy"""
    def __init__(
        self,
        obs_dim: int = 25,
        action_dim: int = 14,
        hidden_dims: List[int] = [256, 256],
        activation=nn.ReLU
    ):
        super().__init__()

        # Build MLP layers
        layers = []
        prev_dim = obs_dim + action_dim  # Concatenate obs and action
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation(),
            ])
            prev_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)
        
        # Output heads
        self.correction_head = nn.Linear(prev_dim, action_dim)
        self.intervention_head = nn.Linear(prev_dim, 1)

    def forward(self, obs, policy_action):
        # Concatenate observation and action
        x = torch.cat([obs, policy_action], dim=-1)
        
        # Shared features
        features = self.shared_net(x)
        
        # Predict correction and intervention
        correction = self.correction_head(features)
        intervention_logits = self.intervention_head(features)
        
        return correction, intervention_logits

class ResidualPolicyModule(pl.LightningModule):
    def __init__(
        self,
        obs_dim: int = 25,
        action_dim: int = 14,
        hidden_dims: List[int] = [256, 256],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        intervention_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.policy = MLPResidualPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )
        self.intervention_loss_weight = intervention_loss_weight
        
    def training_step(self, batch, batch_idx):
        correction_pred, intervention_logits = self.policy(
            batch["obs"], batch["policy_action"]
        )
        
        # Compute intervention loss
        intervention_loss = F.binary_cross_entropy_with_logits(
            intervention_logits,
            batch["is_intervention"]
        )
        
        # Compute correction loss only on intervention steps
        mask = batch["is_intervention"].bool()
        if mask.sum() > 0:
            correction_loss = F.mse_loss(
                correction_pred[mask],
                batch["correction"][mask]
            )
        else:
            correction_loss = torch.tensor(0.0, device=self.device)
            
        # Total loss
        loss = correction_loss + self.intervention_loss_weight * intervention_loss
        
        # Log metrics
        self.log("train/correction_loss", correction_loss, prog_bar=True)
        self.log("train/intervention_loss", intervention_loss, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        correction_pred, intervention_logits = self.policy(
            batch["obs"], batch["policy_action"]
        )
        
        # Compute intervention accuracy
        intervention_pred = (intervention_logits > 0).float()
        intervention_acc = (intervention_pred == batch["is_intervention"]).float().mean()
        
        # Log metrics
        self.log("val/intervention_acc", intervention_acc, prog_bar=True)
        
        # Compute losses same as training for monitoring
        intervention_loss = F.binary_cross_entropy_with_logits(
            intervention_logits,
            batch["is_intervention"]
        )
        
        mask = batch["is_intervention"].bool()
        if mask.sum() > 0:
            correction_loss = F.mse_loss(
                correction_pred[mask],
                batch["correction"][mask]
            )
        else:
            correction_loss = torch.tensor(0.0, device=self.device)
            
        loss = correction_loss + self.intervention_loss_weight * intervention_loss
        
        self.log("val/correction_loss", correction_loss, prog_bar=True)
        self.log("val/intervention_loss", intervention_loss, prog_bar=True)
        self.log("val/total_loss", loss, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

def main():
    # Training settings
    data_dir = "/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/eval/whole_policy/0428-stereo-100_correction/correction_data"  # Update this path
    batch_size = 32
    num_workers = 4
    max_epochs = 100
    
    # Model settings
    obs_dim = 25  # Update based on your observation dimension
    action_dim = 14  # Update based on your action dimension
    hidden_dims = [256, 256]
    lr = 1e-4
    weight_decay = 1e-4
    intervention_loss_weight = 1.0
    
    # Initialize wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"residual_mlp_{timestamp}"
    
    wandb.init(
        project="surrol_residual_policy",
        name=run_name,
        config={
            "batch_size": batch_size,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dims": hidden_dims,
            "lr": lr,
            "weight_decay": weight_decay,
            "intervention_loss_weight": intervention_loss_weight,
        }
    )
    
    # Create datasets
    train_dataset = CorrectionDataset(data_dir, split="train")
    val_dataset = CorrectionDataset(data_dir, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    exit()
    # Create model and trainer
    model = ResidualPolicyModule(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        lr=lr,
        weight_decay=weight_decay,
        intervention_loss_weight=intervention_loss_weight
    )
    
    # Setup checkpoint directory
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        logger=pl.loggers.WandbLogger(),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="{epoch}-{val/intervention_acc:.2f}",
                monitor="val/intervention_acc",
                mode="max",
                save_top_k=3,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]
    )
    
    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    main() 