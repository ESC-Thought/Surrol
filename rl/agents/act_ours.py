import sys
import os 
import torch
import pickle
import torch.nn as nn
import numpy as np
from einops import rearrange
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.append('/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3')
from policy_ours import ACTPolicy, CNNMLPPolicy

class ACTPolicy_Ours(nn.Module):
    def __init__(self, policy_class, config=None):
        super(ACTPolicy_Ours, self).__init__()
        from configs.agent.act_args import args
        args = args()
        lr_backbone = 1e-5
        self.args = args
            
        lr_backbone = 1e-5
        backbone = 'resnet18'
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        
        policy_config = {
            'lr': self.args.lr,
            'num_queries': self.args.chunk_size,
            'kl_weight': self.args.kl_weight,
            'hidden_dim': self.args.hidden_dim,
            'dim_feedforward': self.args.dim_feedforward,
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': self.args.camera_names,
            'images_input': self.args.images_input,
        }
        
        self.policy_class = policy_class
        self.policy_config = policy_config
        self.policy = self._make_policy(policy_class, policy_config)
        self._init_params()
        
        # For skill tracking
        self.skill_history = []
        self.skill_counts = defaultdict(int)
        self.total_timesteps = 0
        
    def _make_policy(self, policy_class, policy_config):
        if policy_class == 'ACT':
            # Add use_stereo flag to detect if we need stereo components
            use_stereo = len(policy_config['camera_names']) >= 2 and 'rgb1' in policy_config['camera_names'] and 'rgb2' in policy_config['camera_names']
            assert use_stereo==True
            # If we're using stereo, make sure the model knows about it
            if use_stereo:
                print('Initializing ACTPolicy with stereo vision support')
                # Make sure the policy config has the right camera names
                if 'rgb1' in policy_config['camera_names'] and 'rgb2' in policy_config['camera_names']:
                    policy_config['use_stereo'] = True
            
            policy = ACTPolicy(policy_config)
            print('Initialized ACTPolicy for evaluation')
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        else:
            raise NotImplementedError(f"Policy class {policy_class} not implemented")
        return policy

    def load_ckpt(self, ckpt_dir, ckpt_name):
        """Load checkpoint and dataset stats"""
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)       
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))   
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded model: {ckpt_path}')
        
        # Load dataset stats for preprocessing
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        self.stats = stats
        return True

    def get_image(self, obs, camera_names_new):
        """Process image observations from environment"""
        curr_images = []
        for cam_name in camera_names_new:
            img = obs['images'][cam_name]
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3, axis=-1)
            curr_image = rearrange(img, 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    def _init_params(self):
        """Initialize tracking variables for inference"""
        self.max_timesteps = self.args.episode_len
        self.state_dim = self.args.state_dim
        self.qpos_history = torch.zeros((1, self.max_timesteps, self.state_dim)).cuda()
        self.query_frequency = self.policy_config['num_queries']
        if self.args.temporal_agg:
            self.query_frequency = 1
            self.num_queries = self.policy_config['num_queries']
        
        # For action aggregation
        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.policy_config['num_queries'], self.state_dim]).cuda()

    def get_action(self, obs, step_idx=0, estimate_uncertainty=False):
        """Get action from policy with optional uncertainty estimation"""
        with torch.inference_mode():
            # Process observation
            qpos_numpy = np.array(obs['qpos'])
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            self.qpos_history[:, step_idx] = qpos
            
            # Prepare camera inputs
            if self.args.images_input is not None:
                camera_names_new = self.args.camera_names + self.args.images_input
            else: 
                camera_names_new = self.args.camera_names
            curr_image = self.get_image(obs, camera_names_new)

            # Get actions from model
            if step_idx % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)
                
                # Track skills if available
                if hasattr(self.policy.model, 'skill_encoder'):
                    with torch.no_grad():
                        _, skill_indices, _ = self.policy.model.skill_encoder(
                            self.policy.model.encoder_output.detach()
                        )
                        skill_idx = skill_indices.item()
                        self.skill_history.append(skill_idx)
                        self.skill_counts[skill_idx] += 1
                        self.total_timesteps += 1
                
            # Apply temporal aggregation if enabled
            if self.args.temporal_agg:
                self.all_time_actions[[step_idx], step_idx:step_idx+self.num_queries] = self.all_actions
                actions_for_curr_step = self.all_time_actions[:, step_idx]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = self.all_actions[:, step_idx % self.query_frequency]

            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            
            return action
    
    def save_skill_visualization(self, save_dir, episode_num=None):
        """Save visualization of skill usage during execution"""
        if not hasattr(self.policy.model, 'skill_encoder') or not self.skill_history:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot skill sequence
        plt.figure(figsize=(15, 3))
        plt.plot(self.skill_history, marker='o')
        plt.title(f'Skill Sequence - Episode {episode_num if episode_num is not None else ""}')
        plt.xlabel('Timestep')
        plt.ylabel('Skill Index')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'skill_sequence_{episode_num if episode_num is not None else "latest"}.png'))
        plt.close()
        
        # Plot skill distribution
        if self.total_timesteps > 0:
            plt.figure(figsize=(10, 5))
            skills = list(range(16))  # Assuming 16 skills
            counts = [self.skill_counts[s] for s in skills]
            plt.bar(skills, [c/self.total_timesteps for c in counts])
            plt.title('Skill Usage Distribution')
            plt.xlabel('Skill Index')
            plt.ylabel('Usage Frequency')
            plt.savefig(os.path.join(save_dir, f'skill_distribution_{episode_num if episode_num is not None else "latest"}.png'))
            plt.close()
            
        # Save to text file
        with open(os.path.join(save_dir, f'skill_stats_{episode_num if episode_num is not None else "latest"}.txt'), 'w') as f:
            f.write('Skill Usage Statistics:\n')
            f.write('-' * 30 + '\n')
            f.write(f'Total timesteps: {self.total_timesteps}\n\n')
            for skill, count in sorted(self.skill_counts.items()):
                f.write(f'Skill {skill}: {count} times ({count/self.total_timesteps:.3f})\n')
    
    def reset_skill_tracking(self):
        """Reset skill tracking between episodes"""
        self.skill_history = []
        self.skill_counts = defaultdict(int)
        self.total_timesteps = 0 