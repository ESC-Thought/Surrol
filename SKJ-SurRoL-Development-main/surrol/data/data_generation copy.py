"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import gym
import time
import numpy as np
import imageio
from surrol.const import ROOT_DIR_PATH
import surrol.gym
import pickle
import cv2
import json

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int, default=None, 
                    help='how many steps allowed to run')
args = parser.parse_args()

class Data():
    def __init__(self):
        self.images = {}
        self.actions = {}
        self.qpos = {}
        self.dq = {}
        self.masks = {}
        self.masks_target = {}
        self.depths = {}
        self.obs = {}
    def save_data(self, episode_idx):
        """
        Save the collected data to a pickle file.
        """
        data = {
            'images': self.images,
            'actions': self.actions,  # TODO 
            'dq_actions': self.dq,  
            'qpos': self.qpos,
            'masks': self.masks,
            'masks_target': self.masks_target,
            'depths': self.depths,
            'observation': self.obs
        }
        # save_path = f'/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/image_action_qpos_{self.episode_idx}.pkl'
        # save_path = f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/collected_data/bipeg/image_action_qpos_{episode_idx}.pkl'
        # save_path = f'/research/d1/gds/kjshi/IROS_SurRoL/data_collected/bi_peg/image_action_qpos_{episode_idx}.pkl'
        save_path = f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/collected_data/bipeg/whole/image_action_qpos_{episode_idx}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved data to {save_path}")

    def reset(self):
        self.images = {}
        self.actions = {}
        self.dq = {}
        self.qpos = {}
        self.masks = {}
        self.masks_target = {}
        self.depths = {}
        self.obs = {}

cnt = 0
success_cnt = 0
def main():
    global cnt
    data_class = Data()

    env = gym.make(args.env, render_mode='human', action_mode = 'rpy')  # 'human', 'rgb_array'  'pitch' for needle regrasp
    num_itr = 1000 if not args.video else 1 # TODO
    init_state_space = 'random'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env._max_episode_steps

    print()
    while success_cnt < 200:
        obs = env.reset()
        
        print("ITERATION NUMBER ", cnt)
        goToGoal(env, obs, data_class)
        cnt += 1
        print('success rate:', success_cnt/cnt)
    
    # save_data()

    used_time = time.time() - init_time
    # print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {cnt}")
    env.close()


def goToGoal(env, last_obs, data_class):
    global success_cnt
    data_class.reset()
    time_step = 0  # count the total number of time steps
    cnt = 0
    waypoint_idx_prev = 0
    waypoint_idx = 0
    episode_init_time = time.time()

    obs, success = last_obs, False
    if args.env == "NeedleRegrasp-v0" or args.env == "BiPegTransfer-v2" :
        action_prev = np.zeros(14)    # 12 for old, 14 for new
    else:
        action_prev = np.zeros(7)     # 6 for old qpos, 7 for new position 
    while time_step < min(env._max_episode_steps, args.steps) and not success:
        action, rgb1, rgb2, mask, depth, robot_state, i = env.get_oracle_action(obs)
        # obs['waypoint'] = waypoint
        if str(waypoint_idx) not in data_class.images.keys():
            data_class.images[f'{str(waypoint_idx)}'] = []
            data_class.actions[f'{str(waypoint_idx)}'] = []
            data_class.dq[f'{str(waypoint_idx)}'] = []
            data_class.qpos[f'{str(waypoint_idx)}'] = []
            data_class.masks[f'{str(waypoint_idx)}'] = []
            data_class.masks_target[f'{str(waypoint_idx)}'] = []
            data_class.depths[f'{str(waypoint_idx)}'] = []
            data_class.obs[f'{str(waypoint_idx)}'] = []

        if args.env == "NeedlePick-v1":   
            if waypoint_idx == 2:
                for i in range(10):
                    data_class.images[f'{str(waypoint_idx)}'].append([np.array(rgb1), np.array(rgb2)])
                    # joint_diffs = [robot_joint_state[i]-action_prev[i]
                    #                 for i in range(len(robot_joint_state))]
                    data_class.actions[f'{str(waypoint_idx)}'].append(action)
                    data_class.dq[f'{str(waypoint_idx)}'].append(joint_diffs)
                    # data_class.qpos[f'{str(waypoint_idx)}'].append(robot_joint_state)
                    # data_class.masks[f'{str(waypoint_idx)}'].append(mask_ori)
                    # data_class.masks_target[f'{str(waypoint_idx)}'].append(mask_target)
                    data_class.depths[f'{str(waypoint_idx)}'].append(depth)
            else:
                data_class.images[f'{str(waypoint_idx)}'].append([np.array(rgb1), np.array(rgb2)])
                # joint_diffs = [robot_joint_state[i]-action_prev[i]
                                # for i in range(len(robot_joint_state))]
                data_class.actions[f'{str(waypoint_idx)}'].append(action)
                data_class.dq[f'{str(waypoint_idx)}'].append(joint_diffs)
                # data_class.qpos[f'{str(waypoint_idx)}'].append(robot_joint_state)
                # data_class.masks[f'{str(waypoint_idx)}'].append(mask_ori)
                # data_class.masks_target[f'{str(waypoint_idx)}'].append(mask_target)
                data_class.depths[f'{str(waypoint_idx)}'].append(depth)
        else:
            # data_class.images[f'{str(waypoint_idx)}'].append([np.array(rgb1), np.array(rgb2)])
            # joint_diffs = [robot_joint_state[i]-action_prev[i]
            #                 for i in range(len(robot_joint_state))]
            # data_class.actions[f'{str(waypoint_idx)}'].append(action)
            # data_class.dq[f'{str(waypoint_idx)}'].append(joint_diffs)
            # data_class.qpos[f'{str(waypoint_idx)}'].append(robot_joint_state)
            # data_class.masks[f'{str(waypoint_idx)}'].append(mask_ori)
            # data_class.masks_target[f'{str(waypoint_idx)}'].append(mask_target)
            data_class.depths[f'{str(waypoint_idx)}'].append(depth)    
            data_class.obs[f'{str(waypoint_idx)}'].append(obs)        
        # if waypoint_idx in [1]:
            # action *= 0.5
        if waypoint_idx_prev != waypoint_idx:
            print(f"cnt to finish {waypoint_idx_prev} is :{cnt}")
            cnt = 0

        obs, reward, done, info = env.step(action)

        # action_prev = robot_joint_state
        time_step += 1
        cnt += 1
        
        waypoint_idx_prev = waypoint_idx
        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True
            # if time_step <100:
            # data_class.save_data(success_cnt)
            success_cnt += 1
            
        time.sleep(0.01)
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))




if __name__ == "__main__":
    main()
