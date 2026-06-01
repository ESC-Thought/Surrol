import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import torchvision.models as models
# from dinov2.models import vision_transformer as vits

class ImageEncoder(nn.Module):
    def __init__(self, encoder_type='resnet18'):
        super(ImageEncoder, self).__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'resnet18':
            # Load pretrained ResNet18
            resnet = models.resnet18(pretrained=True)
            # Remove the final fully connected layer
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.output_dim = 512
            
        elif encoder_type == 'dinov2':
            # Load pretrained DINOv2
            self.encoder = vits.__dict__['vit_small'](patch_size=14, num_classes=0)
            self.encoder.load_state_dict(torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
            ))
            self.output_dim = 384  # DINOv2-S output dimension
            
        # Freeze the backbone
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        if self.encoder_type == 'resnet18':
            return self.encoder(x).squeeze(-1).squeeze(-1)
        else:  # dinov2
            return self.encoder(x)

class CNNEncoder(nn.Module):
    def __init__(self, encoder_type='resnet18'):
        super(CNNEncoder, self).__init__()
        
        # Initialize encoders for each input type
        self.rgb_encoder = ImageEncoder(encoder_type)
        self.seg_encoder = ImageEncoder(encoder_type)  # You might want to modify first layer for 1-channel input
        self.depth_encoder = ImageEncoder(encoder_type)  # You might want to modify first layer for 1-channel input
        
        # Modify the first conv layer of seg and depth encoders to accept 1-channel input
        if encoder_type == 'resnet18':
            # For ResNet
            self.seg_encoder.encoder[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.depth_encoder.encoder[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            # For DINOv2, modify the patch embedding layer
            self.seg_encoder.encoder.patch_embed.proj = nn.Conv2d(1, self.seg_encoder.encoder.embed_dim, 
                                                                kernel_size=14, stride=14)
            self.depth_encoder.encoder.patch_embed.proj = nn.Conv2d(1, self.depth_encoder.encoder.embed_dim, 
                                                                   kernel_size=14, stride=14)
        
        # Calculate total feature dimension
        self.feature_dim = self.rgb_encoder.output_dim * 3  # Concatenating RGB, seg, and depth features
        
    def forward(self, rgb, seg, depth):
        # Get features from each encoder
        rgb_features = self.rgb_encoder(rgb)
        seg_features = self.seg_encoder(seg)
        depth_features = self.depth_encoder(depth)
        
        # Concatenate all features
        features = torch.cat([rgb_features, seg_features, depth_features], dim=1)
        return features

def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)

    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
def build_mlp(
    input_dim,
    hidden_sizes,
    output_dim,
    activation,
    output_std=1.0,
    bias_on_last_layer=True,
    last_layer_bias_const=0.0,
):
    act_func = getattr(nn, activation)
    layers = []
    layers.append(
        layer_init(nn.Linear(input_dim, hidden_sizes[0]), nonlinearity=activation)
    )
    layers.append(act_func())
    for i in range(1, len(hidden_sizes)):
        layers.append(
            layer_init(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nonlinearity=activation
            )
        )
        layers.append(act_func())
    layers.append(
        layer_init(
            nn.Linear(hidden_sizes[-1], output_dim, bias=bias_on_last_layer),
            std=output_std,
            nonlinearity="Tanh",
            bias_const=last_layer_bias_const,
        )
    )
    return nn.Sequential(*layers)

class PPOResidualNetwork(nn.Module):
    def __init__(
        self, 
        action_dim,
        actor_hidden_size=256,
        actor_num_layers=2,
        critic_hidden_size=256,
        critic_num_layers=2,
        init_logstd=-3,
        action_head_std=0.01,
        action_scale=0.1,
        encoder_type='resnet18'
    ):
        super(PPOResidualNetwork, self).__init__()
        
        # Image encoder
        self.encoder = CNNEncoder(encoder_type)
        feature_dim = self.encoder.feature_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        # Add layer normalization after encoder
        self.feature_norm = nn.LayerNorm(feature_dim)

        # Actor network (mean) using build_mlp
        self.actor_mean = build_mlp(
            input_dim=feature_dim,
            hidden_sizes=[actor_hidden_size] * actor_num_layers,
            output_dim=action_dim,
            activation="SiLU",
            output_std=action_head_std,
            bias_on_last_layer=True  # Changed to True for stability
        )

        # Add tanh activation for action mean
        self.action_activation = nn.Tanh()

        # Critic network using build_mlp
        self.critic = build_mlp(
            input_dim=feature_dim,
            hidden_sizes=[critic_hidden_size] * critic_num_layers,
            output_dim=1,
            activation="SiLU",
            output_std=1.0,
            bias_on_last_layer=True,
            last_layer_bias_const=0.0
        )

        # Initialize logstd with a more stable value
        self.actor_logstd = nn.Parameter(
            torch.ones(1, action_dim) * init_logstd,
            requires_grad=True
        )
        
        # Add min/max values for logstd
        self.min_logstd = -20
        self.max_logstd = 2

    def forward(self, rgb, seg, depth):
        features = self.encoder(rgb, seg, depth)
        features = self.feature_norm(features)
        
        # Get action mean and apply tanh
        action_mean = self.actor_mean(features)
        action_mean = self.action_activation(action_mean)
        
        # Clamp logstd for stability
        action_logstd = torch.clamp(
            self.actor_logstd.expand_as(action_mean),
            self.min_logstd,
            self.max_logstd
        )
        action_std = torch.exp(action_logstd)
        
        value = self.critic(features)
        
        return action_mean, action_std, value

    def get_value(self, rgb, seg, depth):
        features = self.encoder(rgb, seg, depth)
        return self.critic(features)

    def get_action_and_value(self, rgb, seg, depth, action=None):
        features = self.encoder(rgb, seg, depth)
        features = self.feature_norm(features)
        
        # Get action mean and apply tanh
        action_mean = self.actor_mean(features)  # Shape: [batch_size, action_dim]
        action_mean = self.action_activation(action_mean)
        
        # Clamp logstd for stability
        action_logstd = torch.clamp(
            self.actor_logstd.expand_as(action_mean),
            self.min_logstd,
            self.max_logstd
        )
        action_std = torch.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.rsample()  # Shape: [batch_size, action_dim]
        
        # Calculate log probs and entropy
        log_prob = probs.log_prob(action)  # Shape: [batch_size, action_dim]
        log_prob = log_prob.sum(dim=-1)  # Shape: [batch_size]
        
        entropy = probs.entropy().sum(dim=-1)  # Shape: [batch_size]
        value = self.critic(features).squeeze(-1)  # Shape: [batch_size]
        
        return (
            action,
            log_prob,  # Already summed across action dimensions
            entropy,
            value,
            action_mean,
        )



class PPOAgent:
    def __init__(
        self, 
        action_dim, 
        device="cuda", 
        lr=3e-4, 
        encoder_type='resnet18',
        action_scale=0.1
    ):
        self.device = device
        self.network = PPOResidualNetwork(
            action_dim=action_dim,
            encoder_type=encoder_type,
            action_scale=action_scale
        ).to(device)
        
        # Use separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(
            list(self.network.encoder.parameters()) + 
            list(self.network.actor_mean.parameters()) + 
            [self.network.actor_logstd],
            lr=lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.network.critic.parameters()),
            lr=lr
        )
        
        # PPO hyperparameters
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5  # Add gradient clipping
        self.action_scale = action_scale
        self.last_log_prob = None  # Add this line to store log probabilities

    def _preprocess_images(self, rgb, seg, depth):
        # Convert numpy arrays to tensors and normalize
        if len(rgb.shape) == 3:  # Single image
            rgb = torch.FloatTensor(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            seg = torch.FloatTensor(seg).unsqueeze(0).unsqueeze(0).to(self.device)
            depth = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(self.device)
        else:  # Batch of images
            rgb = torch.FloatTensor(rgb).permute(0, 3, 1, 2).to(self.device) / 255.0
            seg = torch.FloatTensor(seg).unsqueeze(1).to(self.device)
            depth = torch.FloatTensor(depth).unsqueeze(1).to(self.device)
        return rgb, seg, depth
    
    def get_action(self, obs, base_action):
        with torch.no_grad():
            rgb = obs['images']['rgb1']
            seg = obs['images']['mask']
            depth = obs['images']['depth']
            rgb, seg, depth = self._preprocess_images(rgb, seg, depth)
            
            action, log_prob, _, _, action_mean = self.network.get_action_and_value(rgb, seg, depth)
            
            # Ensure action has correct shape and convert to numpy
            residual_action = action.squeeze().cpu().numpy() * self.action_scale
            
            # Ensure base_action is numpy array with correct shape
            base_action = np.array(base_action).reshape(-1)
            
            # Combine base action with residual and ensure correct shape
            final_action = base_action + residual_action
            
            # Verify action dimensions
            if hasattr(self.network, 'action_dim'):
                assert final_action.shape[0] == self.network.action_dim, \
                    f"Action dimension mismatch. Expected {self.network.action_dim}, got {final_action.shape[0]}"
            
            # Store log probability for later use
            self.last_log_prob = log_prob.cpu().numpy()
            
            return final_action, residual_action
    
    def update(self, obs_batch, action_batch, old_log_probs, returns, advantages):
        batch_size = len(obs_batch)
        
        # Preprocess images and convert to tensors
        rgb_batch = torch.FloatTensor(np.array([obs['images']['rgb1'] for obs in obs_batch])).permute(0, 3, 1, 2).to(self.device) / 255.0
        seg_batch = torch.FloatTensor(np.array([obs['images']['mask'] for obs in obs_batch])).unsqueeze(1).to(self.device)
        depth_batch = torch.FloatTensor(np.array([obs['images']['depth'] for obs in obs_batch])).unsqueeze(1).to(self.device)
        
        # Convert actions and ensure proper shape
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        action_batch = action_batch.reshape(batch_size, -1)  # Shape: [batch_size, action_dim]
        
        # Convert other inputs and ensure proper shapes
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device).reshape(batch_size)  # Shape: [batch_size]
        returns = torch.FloatTensor(returns).to(self.device).reshape(batch_size)  # Shape: [batch_size]
        advantages = torch.FloatTensor(advantages).to(self.device).reshape(batch_size)  # Shape: [batch_size]
        
        # First compute everything we need with a single forward pass
        with torch.no_grad():
            _, _, _, values, _ = self.network.get_action_and_value(
                rgb_batch, seg_batch, depth_batch, action_batch
            )
        
        # Update critic first
        self.critic_optimizer.zero_grad()
        _, _, _, new_values, _ = self.network.get_action_and_value(
            rgb_batch, seg_batch, depth_batch, action_batch
        )
        value_loss = 0.5 * (returns - new_values.reshape(batch_size)).pow(2).mean()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Then update actor
        self.actor_optimizer.zero_grad()
        _, new_log_probs, entropy, _, action_mean = self.network.get_action_and_value(
            rgb_batch, seg_batch, depth_batch, action_batch
        )
        
        # Calculate policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)  # Shape: [batch_size]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add L1/L2 regularization for residual actions
        residual_l1_loss = torch.mean(torch.abs(action_mean))
        residual_l2_loss = torch.mean(torch.square(action_mean))
        
        # Compute actor loss
        actor_loss = (
            policy_loss - 
            self.entropy_coef * entropy.mean() +
            0.01 * residual_l1_loss +
            0.01 * residual_l2_loss
        )
        
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.mean().item() 