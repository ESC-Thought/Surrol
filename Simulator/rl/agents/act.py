import sys
import os 

sys.path.append('/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3')
from policy import ACTPolicy, CNNMLPPolicy
import yaml
import torch
import pickle
import torch.nn as nn
import numpy as np
from einops import rearrange
sys.path.append('/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
        print('ACTPolicy')
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

class ACT_Policy(nn.Module):
    def __init__(self, policy_class):
        super(ACT_Policy, self).__init__()
        from Simulator.rl.configs.agent.act_args import args
        args = args()
        lr_backbone = 1e-5
        self.args = args
        backbone = 'resnet18'
        if policy_class == 'ACT':
            enc_layers = 4
            dec_layers = 7
            nheads = 8
            policy_config = {'lr': args.lr,
                                'num_queries': args.chunk_size,
                                'kl_weight': args.kl_weight,
                                'hidden_dim': args.hidden_dim,
                                'dim_feedforward': args.dim_feedforward,
                                'lr_backbone': lr_backbone,
                                'backbone': backbone,
                                'enc_layers': enc_layers,
                                'dec_layers': dec_layers,
                                'nheads': nheads,
                                'camera_names': args.camera_names,   # input camera_names to the model
                                'images_input': args.images_input,
                                'policy_class': 'ACT',
                                'seed': args.seed,
                                'num_epochs': args.num_epochs,
                                'task_name': args.task_name,
                                'ckpt_dir': args.ckpt_dir,
                                }
        else:
            raise NotImplementedError
        self.policy_class = policy_class
        self.policy_config = policy_config
        self.policy = make_policy(policy_class, policy_config)
        self._init_params()
    def load_ckpt(self, ckpt_dir, ckpt_name):
         # load policy and stats
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)       # 加载指定的策略模型检查点ckpt_name
        # 使用 .load_state_dict 将预训练模型权重加载到模型中，并将模型设置为评估模式(eval())
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))   
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')
        
        # 加载预处理数据的统计信息（如均值、标准差），用于后续动作和状态的标准化。
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        # 定义状态和动作的标准化与去标准化函数，用于确保状态和动作的数据尺度与模型训练时一致。
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    def get_image(self, ts, camera_names_new):
        curr_images = []
        for cam_name in camera_names_new:
            # print("cam name:", cam_name)
            # sky
            # curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            img = ts['images'][cam_name]
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)  # Expand to (256, 256, 1)
                img = np.repeat(img, 3, axis=-1)    # Repeat along the channel axis to make (256, 256, 3)
        
            # curr_image = rearrange(ts['images'][cam_name], 'h w c -> c h w')
            curr_image = rearrange(img, 'h w c -> c h w')
            
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    def _init_params(self):
        self.max_timesteps = self.args.episode_len
        self.state_dim = self.args.state_dim
        self.qpos_history = torch.zeros((1, self.max_timesteps, self.state_dim)).cuda()
        self.image_list = [] # for visualization
        self.qpos_list = []
        self.target_qpos_list = []
        self.query_frequency = self.policy_config['num_queries']
        if self.args.temporal_agg:
            self.query_frequency = 1
            self.num_queries = self.policy_config['num_queries']


    def get_action(self, obs, step_idx=0, estimate_uncertainty=False):
        """Get action with optional uncertainty estimation using MC dropout"""
        with torch.inference_mode():
            # Process observation
            qpos_numpy = np.array(obs['qpos'])
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            self.qpos_history[:, step_idx] = qpos
            if self.args.images_input is not None:
                camera_names_new = self.args.camera_names + self.args.images_input
            else: 
                camera_names_new = self.args.camera_names
            curr_image = self.get_image(obs, camera_names_new)

            if estimate_uncertainty:
                # Enable dropout for uncertainty estimation
                self.policy.eval()
                self.policy.enable_dropout()
                
                # Perform multiple forward passes
                n_samples = 10
                action_samples = []
                contact_pos_samples = []
                
                for _ in range(n_samples):
                    if step_idx % self.query_frequency == 0:
                        actions= self.policy(qpos, curr_image)
                        
                    if self.args.temporal_agg:
                        all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).cuda()
                        all_time_actions[[step_idx], step_idx:step_idx+self.num_queries] = actions
                        actions_for_curr_step = all_time_actions[:, step_idx]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = actions[:, step_idx % self.query_frequency]
                    
                    action_samples.append(raw_action)
                    # contact_pos_samples.append(contact_pred)
                
                # Stack and compute statistics
                action_samples = torch.stack(action_samples)
                # contact_pos_samples = torch.stack(contact_pos_samples)
                
                action_mean = action_samples.mean(dim=0)
                action_std = action_samples.std(dim=0)
                # contact_pos_mean = contact_pos_samples.mean(dim=0)
                # contact_pos_std = contact_pos_samples.std(dim=0)
                
                # Post-process mean action
                raw_action = action_mean.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                
                return {
                    'action': action,
                    'action_mean': action_mean.cpu().numpy(),
                    'action_std': action_std.cpu().numpy(),
                    'contact_pos_mean': contact_pos_mean.cpu().numpy(),
                    'contact_pos_std': contact_pos_std.cpu().numpy()
                }
                
            else:
                # Regular point prediction without uncertainty
                if step_idx % self.query_frequency == 0:
                    self.all_actions= self.policy(qpos, curr_image)
                    
                if self.args.temporal_agg:
                    all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).cuda()
                    all_time_actions[[step_idx], step_idx:step_idx+self.num_queries] = self.all_actions
                    actions_for_curr_step = all_time_actions[:, step_idx]
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

if __name__ == '__main__':
    act_class = ACT_Policy('ACT')
    act_class.load_ckpt('/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3/experiments20250212_2258_bi_peg_transfer', 'policy_best.ckpt')

    exit()