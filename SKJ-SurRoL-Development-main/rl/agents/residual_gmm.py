import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import torchvision.models as models
from collections import deque
import torch.nn.functional as F

# StereoEncoder using direct torch.hub loading
class StereoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pretrained DINOv2 directly from hub
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.output_dim = 384
        
        # Modify first layer for 6-channel stereo input
        original_weight = self.encoder.patch_embed.proj.weight
        new_weight = torch.zeros(384, 6, 14, 14, device=original_weight.device)
        new_weight[:, :3, :, :] = original_weight
        new_weight[:, 3:, :, :] = original_weight
        self.encoder.patch_embed.proj = nn.Conv2d(6, 384, kernel_size=14, stride=14)
        self.encoder.patch_embed.proj.weight = nn.Parameter(new_weight)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):  # x: [B, 6, H, W]
        return self.encoder(x)  # [B, 384]

class SkillEncoder(nn.Module):
    def __init__(self, obs_dim=384, single_action_dim=3, chunk_size=5, skill_dim=32):
        super().__init__()
        self.chunk_size = chunk_size
        action_chunk_dim = single_action_dim * chunk_size
        
        # Temporal convolution for action chunk processing
        self.action_encoder = nn.Sequential(
            nn.Linear(action_chunk_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Final skill encoder
        self.skill_net = nn.Sequential(
            nn.Linear(obs_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, skill_dim)
        )
    
    def forward(self, z_obs, action_chunk):  # z_obs: [B, 384], action_chunk: [B, chunk_size * 3]
        # Process action chunk
        action_features = self.action_encoder(action_chunk)
        
        # Combine with visual features
        x = torch.cat([z_obs, action_features], dim=1)
        return self.skill_net(x)  # [B, 32]

class ResidualNetwork(nn.Module):
    def __init__(self, obs_dim=384, single_action_dim=3, skill_dim=32, dropout_p=0.1):
        super().__init__()
        # Current and next action (3 + 3), plus obs and skill
        input_dim = obs_dim + (2 * single_action_dim) + skill_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, single_action_dim)  # Output delta for current action only
        )
        self.dropout_p = dropout_p
        
    def forward(self, z_obs, current_act, next_act, skill):
        x = torch.cat([z_obs, current_act, next_act, skill], dim=1)
        return self.net(x)

class ResidualAgent:
    def __init__(self, single_action_dim=3, chunk_size=5, device="cuda", lr=1e-4):
        self.device = device
        self.single_action_dim = single_action_dim
        self.chunk_size = chunk_size
        
        # Initialize networks
        self.visual_encoder = StereoEncoder().to(device)
        self.skill_encoder = SkillEncoder(
            obs_dim=self.visual_encoder.output_dim,
            single_action_dim=single_action_dim,
            chunk_size=chunk_size
        ).to(device)
        self.residual_net = ResidualNetwork(
            obs_dim=self.visual_encoder.output_dim,
            single_action_dim=single_action_dim
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.skill_encoder.parameters()) + 
            list(self.residual_net.parameters()),
            lr=lr
        )
        
        # Initialize correction buffer and metrics
        self.buffer_size = 50
        self.correction_buffer = deque(maxlen=self.buffer_size)
        self.metrics = {
            'total_corrections': 0,
            'success_rate': deque(maxlen=100),
            'uncertainty_history': deque(maxlen=100)
        }
        
    def _preprocess_stereo(self, left_rgb, right_rgb):
        # Convert to PyTorch tensors
        left_tensor = torch.FloatTensor(left_rgb).permute(2,0,1)
        right_tensor = torch.FloatTensor(right_rgb).permute(2,0,1)
        
        # Resize to standard ViT dimensions (224×224)
        target_h, target_w = 224, 224
        
        left_tensor = F.interpolate(
            left_tensor.unsqueeze(0), 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        right_tensor = F.interpolate(
            right_tensor.unsqueeze(0), 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Combine and normalize stereo images
        stereo = torch.cat([left_tensor, right_tensor], dim=0).unsqueeze(0).to(self.device) / 255.0
        return stereo
    
    @torch.no_grad()
    def get_action(self, obs, act_chunk, current_step, n_samples=10, uncertainty_threshold=0.1):
        """Predict residual correction with uncertainty for current action"""
        # Process stereo input
        stereo = self._preprocess_stereo(
            obs['images']['rgb1'],
            obs['images']['rgb2']
        )
        
        # Get visual embedding
        z_obs = self.visual_encoder(stereo)
        
        # Convert action chunk to tensor
        act_chunk = torch.FloatTensor(act_chunk).to(self.device)
        if act_chunk.dim() == 1:
            act_chunk = act_chunk.unsqueeze(0)
            
        # Get current and next action from chunk
        current_act = act_chunk[:, current_step*self.single_action_dim:(current_step+1)*self.single_action_dim]
        next_act = act_chunk[:, (current_step+1)*self.single_action_dim:(current_step+2)*self.single_action_dim]
        
        # Get skill embedding using full chunk
        skill = self.skill_encoder(z_obs, act_chunk)
        
        # Multiple forward passes with dropout
        self.residual_net.train()  # Enable dropout
        corrections = []
        for _ in range(n_samples):
            delta = self.residual_net(
                z_obs, 
                current_act,
                next_act,
                skill
            )
            corrections.append(delta)
        
        # Compute statistics
        corrections = torch.stack(corrections, dim=0)
        mean_correction = corrections.mean(dim=0)
        uncertainty = corrections.std(dim=0)
        
        self.residual_net.eval()
        
        # Store uncertainty for metrics
        self.metrics['uncertainty_history'].append(uncertainty.mean().item())
        
        # Skip correction if uncertainty is too high
        if uncertainty.mean().item() > uncertainty_threshold:
            return (
                current_act.squeeze(0).cpu().numpy(),  # Original action
                torch.zeros_like(mean_correction).squeeze(0).cpu().numpy(),  # No correction
                uncertainty.squeeze(0).cpu().numpy()
            )
        
        # Combine with base action
        final_action = current_act + mean_correction
        
        return (
            final_action.squeeze(0).cpu().numpy(),
            mean_correction.squeeze(0).cpu().numpy(),
            uncertainty.squeeze(0).cpu().numpy()
        )
    
    def update(self, obs, act_chunk, current_step, correction):
        """Online few-shot update from a single correction"""
        # Add to buffer
        self.correction_buffer.append((obs, act_chunk, current_step, correction))
        self.metrics['total_corrections'] += 1
        
        # Perform 3 gradient steps
        losses = []
        for _ in range(3):
            total_loss = 0
            self.optimizer.zero_grad()
            
            # Process all samples in buffer
            for obs_i, chunk_i, step_i, corr_i in self.correction_buffer:
                # Preprocess data
                stereo = self._preprocess_stereo(
                    obs_i['images']['rgb1'],
                    obs_i['images']['rgb2']
                )
                chunk_i = torch.FloatTensor(chunk_i).to(self.device)
                if chunk_i.dim() == 1:
                    chunk_i = chunk_i.unsqueeze(0)
                corr_i = torch.FloatTensor(corr_i).to(self.device)
                
                # Get current and next action
                current_act = chunk_i[:, step_i*self.single_action_dim:(step_i+1)*self.single_action_dim]
                next_act = chunk_i[:, (step_i+1)*self.single_action_dim:(step_i+2)*self.single_action_dim]
                
                # Forward pass
                z_obs = self.visual_encoder(stereo)
                skill = self.skill_encoder(z_obs, chunk_i)
                pred_correction = self.residual_net(
                    z_obs,
                    current_act,
                    next_act,
                    skill
                )
                
                # MSE loss
                loss = F.mse_loss(pred_correction.squeeze(0), corr_i)
                total_loss += loss
            
            # Update
            total_loss.backward()
            self.optimizer.step()
            losses.append(total_loss.item())
        
        return np.mean(losses)
    
    def log_success(self, success):
        """Log success/failure for metrics"""
        self.metrics['success_rate'].append(float(success))
    
    def get_metrics(self):
        """Return current performance metrics"""
        return {
            'success_rate': np.mean(list(self.metrics['success_rate'])) if self.metrics['success_rate'] else 0.0,
            'total_corrections': self.metrics['total_corrections'],
            'mean_uncertainty': np.mean(list(self.metrics['uncertainty_history'])) if self.metrics['uncertainty_history'] else 0.0
        } 