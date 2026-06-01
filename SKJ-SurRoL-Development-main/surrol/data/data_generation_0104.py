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

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
args = parser.parse_args()

class Data():
    def __init__(self):
        self.images = {}
        self.actions = {}
        self.new_actions ={}  # sky
        self.qpos = {}
        self.masks = {}
        self.depths = {}
    def save_data(self, episode_idx):
        """
        Save the collected data to a pickle file.
        """
        data = {
            'images': self.images,
            'actions': self.actions,  # TODO 
            'dq_actions': self.dq,  # sky
            # 'actions': self.new_actions,
            'qpos': self.qpos,
            'masks': self.masks,
            'depths': self.depths
        }
        # save_path = f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/collected_data/needle_pick/image_action_qpos_{episode_idx}.pkl'
        # save_path = f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/0107/needle_pick/image_action_qpos_{episode_idx}.pkl'
        save_path = f'/home/skylar/SurRoL/IROS_SurRoL/rl/act-main-3/data/0108-1/image_action_qpos_{episode_idx}.pkl'
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved data to {save_path}")

    def reset(self):
        self.images = {}
        self.actions = {}
        self.qpos = {}
        self.dq = {}
        self.masks = {}
        self.depths = {}

cnt = 0
def main():
    global cnt
    data_class = Data()

    env = gym.make(args.env, render_mode='rgb_array')  # 'human'
    num_itr = 1000 if not args.video else 1 # TODO
    init_state_space = 'random'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env._max_episode_steps

    print()
    while cnt < 100:
        obs = env.reset()
        
        print("ITERATION NUMBER ", cnt)
        goToGoal(env, obs, data_class)
        cnt += 1
    
    # save_data()

    used_time = time.time() - init_time
    # print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {cnt}")
    env.close()


def goToGoal(env, last_obs, data_class):

    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()

    obs, success = last_obs, False

    while time_step < min(env._max_episode_steps, args.steps) and not success:
        action, rgb1, rgb2, mask, depth, robot_joint_state, waypoint_idx = env.get_oracle_action(obs)

        if str(waypoint_idx) not in data_class.images.keys():
            data_class.images[f'{str(waypoint_idx)}'] = []
            data_class.actions[f'{str(waypoint_idx)}'] = []
            data_class.dq[f'{str(waypoint_idx)}'] = []
            data_class.qpos[f'{str(waypoint_idx)}'] = []
            data_class.masks[f'{str(waypoint_idx)}'] = []
            data_class.depths[f'{str(waypoint_idx)}'] = []
        
        if time_step > 0:
            data_class.images[f'{str(waypoint_idx)}'].append([np.array(rgb1), np.array(rgb2)])
            # cv2.imshow('mask', rgb1)
            # cv2.waitKey(1)
            joint_diffs = [robot_joint_state[i]-action_prev[i]
                        for i in range(len(robot_joint_state))]
            
            data_class.actions[f'{str(waypoint_idx)}'].append(action)
            data_class.dq[f'{str(waypoint_idx)}'].append(joint_diffs)
            data_class.qpos[f'{str(waypoint_idx)}'].append(robot_joint_state)
            data_class.masks[f'{str(waypoint_idx)}'].append(mask)
            data_class.depths[f'{str(waypoint_idx)}'].append(depth)

        action_prev = robot_joint_state 
        obs, reward, done, info = env.step(action)

        time_step += 1

        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True
            data_class.save_data(cnt)
            data_class.reset()
        time.sleep(0.01)
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))




if __name__ == "__main__":
    main()
