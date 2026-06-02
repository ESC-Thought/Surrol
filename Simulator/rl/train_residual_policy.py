import torch
import numpy as np
from Simulator.rl.agents.residual_gmm import ResidualAgent
from Simulator.rl.agents.act import ACT_Policy
import os
import tqdm
from collections import deque
from surrol.tasks.peg_transfer_bimanual_new import BiPegTransfer
import cv2
from datetime import datetime
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

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
        
        # Split train/val
        np.random.shuffle(self.episode_paths)
        split_idx = int(len(self.episode_paths) * 0.9)
        self.episode_paths = self.episode_paths[:split_idx] if split == "train" else self.episode_paths[split_idx:]
        
        # Load all data
        self.data = {
            "is_human_intervention": [],
            "policy_action": [],
            "policy_obs": [],
            "correction_action": [],
            "images": []
        }
        
        for path in self.episode_paths:
            with h5py.File(path, 'r') as f:
                for key in self.data.keys():
                    if key == "images":
                        # Handle image data specially if needed
                        continue
                    self.data[key].extend(f[key][()])
                    
        # Convert to numpy arrays
        for key in self.data.keys():
            if key != "images":
                self.data[key] = np.array(self.data[key])

    def __len__(self):
        return len(self.data["policy_obs"])

    def __getitem__(self, idx):
        return {
            "is_intervention": self.data["is_human_intervention"][idx],
            "policy_action": self.data["policy_action"][idx],
            "obs": self.data["policy_obs"][idx],
            "correction": self.data["correction_action"][idx]
        }

class ResidualPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int = 25,
        action_dim: int = 14,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.correction_head = nn.Linear(hidden_dim, action_dim)
        self.intervention_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, policy_action):
        obs_feat = self.obs_encoder(obs)
        action_feat = self.action_encoder(policy_action)
        
        # Concatenate features
        combined = torch.cat([obs_feat, action_feat], dim=-1)
        features = self.fusion(combined)
        
        # Predict correction and intervention
        correction = self.correction_head(features)
        intervention_logits = self.intervention_head(features)
        
        return correction, intervention_logits

class ResidualPolicyModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        intervention_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.policy = ResidualPolicy()
        self.intervention_loss_weight = intervention_loss_weight
        
    def training_step(self, batch, batch_idx):
        correction_pred, intervention_logits = self.policy(
            batch["obs"], batch["policy_action"]
        )
        
        # Compute intervention loss
        intervention_loss = F.binary_cross_entropy_with_logits(
            intervention_logits.squeeze(-1),
            batch["is_intervention"].float()
        )
        
        # Compute correction loss only on intervention steps
        mask = batch["is_intervention"] == 1
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
        self.log("train/correction_loss", correction_loss)
        self.log("train/intervention_loss", intervention_loss)
        self.log("train/total_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        correction_pred, intervention_logits = self.policy(
            batch["obs"], batch["policy_action"]
        )
        
        # Compute intervention accuracy
        intervention_pred = (intervention_logits.squeeze(-1) > 0).float()
        intervention_acc = (intervention_pred == batch["is_intervention"].float()).float().mean()
        
        # Log metrics
        self.log("val/intervention_acc", intervention_acc)
        
        # Compute losses same as training for monitoring
        intervention_loss = F.binary_cross_entropy_with_logits(
            intervention_logits.squeeze(-1),
            batch["is_intervention"].float()
        )
        
        mask = batch["is_intervention"] == 1
        if mask.sum() > 0:
            correction_loss = F.mse_loss(
                correction_pred[mask],
                batch["correction"][mask]
            )
        else:
            correction_loss = torch.tensor(0.0, device=self.device)
            
        loss = correction_loss + self.intervention_loss_weight * intervention_loss
        
        self.log("val/correction_loss", correction_loss)
        self.log("val/intervention_loss", intervention_loss)
        self.log("val/total_loss", loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

def main():
    # Training settings
    data_dir = "path/to/0428-stereo-100_correction"
    batch_size = 32
    num_workers = 4
    max_epochs = 100
    
    # Initialize wandb
    wandb.init(project="surrol_residual_policy")
    
    # Create data module
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
    
    # Create model and trainer
    model = ResidualPolicyModule()
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        logger=pl.loggers.WandbLogger(),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val/intervention_acc",
                mode="max",
                save_top_k=3,
            )
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