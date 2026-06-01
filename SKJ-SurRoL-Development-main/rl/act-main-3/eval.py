import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import pickle
from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from constants import action_mode
from utils_ import load_data # data functions
from utils_ import sample_box_pose, sample_insertion_pose # robot functions
from utils_ import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE
# from surrol.tasks.needle_pick import NeedlePick
from surrol.tasks.needle_pick_kj_edition import NeedlePick
# from surrol.tasks.peg_transfer import PegTransfer
# from surrol.robots.psm import Psm


import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    # if is_sim:
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    # else:
    #     from aloha_scripts.constants import TASK_CONFIGS
    #     task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    images_input = task_config['images_input']
    # camera_names = 'top'

    # fixed parameters
    # sky
    state_dim = 6 #this is the dim size of qpos not TODO
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,   # input camera_names to the model
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'images_input': images_input,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


# def get_image(ts, camera_names):
#     curr_images = []
#     for cam_name in camera_names:
#         # sky
#         # curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
#         curr_image = rearrange(ts['images'][cam_name], 'h w c -> c h w')
        
#         curr_images.append(curr_image)
#     curr_image = np.stack(curr_images, axis=0)
#     curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
#     return curr_image

def get_image(ts, camera_names_new):
    curr_images = []
    for cam_name in camera_names_new:
        print("cam name:", cam_name)
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

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    images_input = config['images_input']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)       # 加载指定的策略模型检查点ckpt_name
    policy = make_policy(policy_class, policy_config)   # make_policy 根据 policy_class 创建策略模型实例
    # 使用 .load_state_dict 将预训练模型权重加载到模型中，并将模型设置为评估模式(eval())
    loading_status = policy.load_state_dict(torch.load(ckpt_path))   
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    
    # 加载预处理数据的统计信息（如均值、标准差），用于后续动作和状态的标准化。
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    # 定义状态和动作的标准化与去标准化函数，用于确保状态和动作的数据尺度与模型训练时一致。
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    env = NeedlePick(render_mode='human',action_mode=action_mode) 
    # env = PegTransfer(render_mode='human',action_mode=action_mode) 
    env_max_reward = 3  

    # 设置策略模型的查询频率，即每隔多少时间步调用一次策略模型生成动作。
    query_frequency = policy_config['num_queries']
    # 如果 temporal_agg 为 True，则设置为 1，以实现时间聚合策略。
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    # evironment and reward evaluation
    # 定义 num_rollouts 作为评估次数，重复执行多次实验以获得更加稳定的评估结果。
    # episode_returns 和 highest_rewards 用于记录每次实验的总回报和最高奖励。
    num_rollouts = 15
    episode_returns = []
    highest_rewards = []
    # 对于每次 rollout:
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        env.reset()
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        # 动作生成与执行
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            # 在每个时间步的循环中：
            for t in range(max_timesteps):
                obs = env._get_obs()
                # obs = ts.observa
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                # 对关节位置进行标准化（pre_process）并保存到 qpos_history。
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                # 从观察状态中提取图像，并转换为模型需要的格式。
                # print("camera_names is:", camera_names)
                camera_names_new = camera_names + images_input
                curr_image = get_image(obs, camera_names_new)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                # 执行动作并收集奖励
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ts_obs, ts_reward, ts_done, ts_info = env.step(action*0.5)
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts_reward)
                if ts_info['is_success']:
                    ckpt_path = "/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/20250117_1905_needle_pick_cs5/policy_best.ckpt"       # 加载指定的策略模型检查点ckpt_name
                    policy = make_policy(policy_class, policy_config)   # make_policy 根据 policy_class 创建策略模型实例
                    # 使用 .load_state_dict 将预训练模型权重加载到模型中，并将模型设置为评估模式(eval())
                    loading_status = policy.load_state_dict(torch.load(ckpt_path))   
                    print(loading_status)
                    policy.cuda()
                    policy.eval()
                    print(f'Loaded: {ckpt_path}')
            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    # 计算并打印评估的成功率和平均回报
    # 统计每个奖励级别以上的出现次数及其比例
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
