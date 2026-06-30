"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import sys
# software render mode
os.environ['PYBULLET_EGL_DEVICE'] = '-1'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
sys.path.insert(0, '/home/escthought/Surrol/')
sys.path.insert(0, '/home/escthought/Surrol/Simulator')

import argparse
import gym
import time
import numpy as np
import imageio
from Simulator.surrol.const import ROOT_DIR_PATH
import Simulator.surrol.gym
import pickle
import cv2
import json
# import h5py

SUCCESS_DATA_DIR = '/home/escthought/Surrol/collected_data/test'
FAILED_PSM2_DATA_DIR = '/home/escthought/Surrol/collected_data/test/failed'

# SUCCESS_DATA_DIR = '/home/escthought/Surrol/collected_data/peg_block_pick_psm2_v3'
# FAILED_PSM2_DATA_DIR = '/home/escthought/Surrol/collected_data/peg_block_pick_psm2_v3/failed'

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int, default=None, 
                    help='how many steps allowed to run')
parser.add_argument('--randomize-block-initial-position', dest='randomize_block_initial_position',
                    action='store_true', default=None,
                    help='randomize BiPegTransfer block initial peg')
parser.add_argument('--no-randomize-block-initial-position', dest='randomize_block_initial_position',
                    action='store_false',
                    help='disable BiPegTransfer block initial peg randomization')
parser.add_argument('--block-initial-pegs', type=str, default=None,
                    help='comma-separated BiPegTransfer source peg link ids, e.g. 6,7,8,9,10,11')
parser.add_argument('--block-initial-xy-jitter', type=float, default=None,
                    help='small XY jitter applied to the initial BiPegTransfer block pose')
args = parser.parse_args()


def build_env_kwargs():
    if args.env != "BiPegTransfer-v4":
        return {}
    env_kwargs = {}
    if args.randomize_block_initial_position is not None:
        env_kwargs["randomize_block_initial_position"] = args.randomize_block_initial_position
    if args.block_initial_pegs is not None:
        env_kwargs["block_initial_peg_indices"] = args.block_initial_pegs
    if args.block_initial_xy_jitter is not None:
        env_kwargs["block_initial_xy_jitter"] = args.block_initial_xy_jitter
    return env_kwargs

class Data():
    def __init__(self):
        self.images = {}
        self.actions = {}
        self.qpos = {}
        self.dq = {}
        # self.masks = {}
        self.masks_no_arm = {}
        self.masks_target = {}
        self.foveal_block = {}
        # self.foveal_tip1 = {}
        # self.foveal_tip2 = {}
        self.recorded_positions = {}
        self.depths = {}
        # self.contact_pose = {}
    def _to_payload(self):
        """
        Build the collected data payload.
        """
        return {
            'images': self.images,
            'actions': self.actions,  # TODO 
            'dq_actions': self.dq,  
            'qpos': self.qpos,
            'masks': self.masks,
            'masks_no_arm': self.masks_no_arm,
            'masks_target': self.masks_target,
            'foveal_block': self.foveal_block,
            # 'foveal_tip1': self.foveal_tip1,
            # 'foveal_tip2': self.foveal_tip2,
            'recorded_positions': self.recorded_positions,
            'depths': self.depths,
            # 'contacts': self.contact_pose
        }

    def _save_payload(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self._to_payload(), f)
        print(f"Saved data to {save_path}")

    def save_data(self, episode_idx):
        """
        Save a successful demonstration to a pickle file.
        """
        # save_path = f'/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/image_action_qpos_{self.episode_idx}.pkl'
        # save_path = f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/collected_data/bipeg/image_action_qpos_{episode_idx}.pkl'
        # save_path = f'/research/d1/gds/kjshi/IROS_SurRoL/data_collected/bi_peg/image_action_qpos_{episode_idx}.pkl'
        # save_path = f'/home/escthought/Surrol/collected_data/bipeg_transfer/image_action_qpos_{episode_idx}.pkl'
        save_path = os.path.join(SUCCESS_DATA_DIR, f'image_action_qpos_{episode_idx}.pkl')
        self._save_payload(save_path)

    def save_failed_data(self, episode_idx):
        """
        Save a failed PegBlockPickPsm2 rollout for debugging.
        """
        save_path = os.path.join(FAILED_PSM2_DATA_DIR, f'image_action_qpos_failed_{episode_idx}.pkl')
        self._save_payload(save_path)

    def reset(self):
        self.images = {}
        self.actions = {}
        self.dq = {}
        self.qpos = {}
        self.masks = {}
        self.masks_no_arm = {}
        self.masks_target = {}
        self.foveal_block = {}
        # self.foveal_tip1 = {}
        # self.foveal_tip2 = {}
        self.recorded_positions = {}
        self.depths = {}
        # self.contact_pose = {}

cnt = 0
success_cnt = 0
def main():
    global cnt
    data_class = Data()
    print("--- Prepare for creating env ---")
    env = gym.make(args.env, render_mode='rgb_array', action_mode='rpy', **build_env_kwargs())  # 'human', 'rgb_array'  'pitch' for needle regrasp
    print("--- Successfully created env ---")
    
    num_itr = 1000 if not args.video else 1 # TODO
    init_state_space = 'random'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env._max_episode_steps

    print()
    while success_cnt < 80:
        obs = env.reset()
        
        print("ITERATION NUMBER ", cnt)
        goToGoal(env, obs, data_class, cnt)
        cnt += 1
        print('success rate:', success_cnt/cnt)
    
    # save_data()

    used_time = time.time() - init_time
    # print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {cnt}")
    env.close()


def goToGoal(env, last_obs, data_class, episode_idx):
    global success_cnt
    data_class.reset()
    time_step = 0  # count the total number of time steps
    cnt = 0
    waypoint_idx_prev = 0
    waypoint_idx = 0
    episode_init_time = time.time()

    obs, success = last_obs, False
    if args.env == "NeedleRegrasp-v0" or args.env == "PegBlockPick-v0" or args.env == "BiPegTransfer-v2" or args.env == "BiPegTransfer-v3" or args.env == "BiPegTransfer-v4":
        action_prev = np.zeros(14)    # 12 for old, 14 for new
    else:
        action_prev = np.zeros(7)     # 6 for old qpos, 7 for new position 
    while time_step < min(env._max_episode_steps, args.steps) and not success:
        action, rgb1, rgb2, mask_ori, mask_no_arm, mask_target, depth, robot_joint_state, waypoint_idx, tip1,tip2,block, recorded_positions = env.get_oracle_action(obs)
        # cv2.imwrite(os.path.join(ROOT_DIR_PATH, 'data', 'rgb1.png'), rgb1)
        # cv2.imwrite(os.path.join(ROOT_DIR_PATH, 'data', 'rgb2.png'), rgb2)
        assert rgb1.shape[0] == 256
        if str(waypoint_idx) not in data_class.images.keys():
            data_class.images[f'{str(waypoint_idx)}'] = []
            data_class.actions[f'{str(waypoint_idx)}'] = []
            data_class.dq[f'{str(waypoint_idx)}'] = []
            data_class.qpos[f'{str(waypoint_idx)}'] = []
            data_class.masks_no_arm[f'{str(waypoint_idx)}'] = []
            data_class.masks_target[f'{str(waypoint_idx)}'] = []
            data_class.foveal_block[f'{str(waypoint_idx)}'] = []
            data_class.depths[f'{str(waypoint_idx)}'] = []
            # data_class.foveal_tip1[f'{str(waypoint_idx)}'] = []
            # data_class.foveal_tip2[f'{str(waypoint_idx)}'] = []
            data_class.recorded_positions[f'{str(waypoint_idx)}'] = []

        if args.env == "NeedlePick-v1":   
            if waypoint_idx == 2:
                for i in range(10):
                    data_class.images[f'{str(waypoint_idx)}'].append([np.array(rgb1), np.array(rgb2)])
                    joint_diffs = [robot_joint_state[i]-action_prev[i]
                                    for i in range(len(robot_joint_state))]
                    data_class.actions[f'{str(waypoint_idx)}'].append(action)
                    data_class.dq[f'{str(waypoint_idx)}'].append(joint_diffs)
                    data_class.qpos[f'{str(waypoint_idx)}'].append(robot_joint_state)
                    data_class.masks_no_arm[f'{str(waypoint_idx)}'].append(mask_no_arm)
                    data_class.masks_target[f'{str(waypoint_idx)}'].append(mask_target)
            else:
                data_class.images[f'{str(waypoint_idx)}'].append([np.array(rgb1), np.array(rgb2)])
                joint_diffs = [robot_joint_state[i]-action_prev[i]
                                for i in range(len(robot_joint_state))]
                data_class.actions[f'{str(waypoint_idx)}'].append(action)
                data_class.dq[f'{str(waypoint_idx)}'].append(joint_diffs)
                data_class.qpos[f'{str(waypoint_idx)}'].append(robot_joint_state)
                data_class.masks_no_arm[f'{str(waypoint_idx)}'].append(mask_no_arm)
                data_class.masks_target[f'{str(waypoint_idx)}'].append(mask_target)
                data_class.depths[f'{str(waypoint_idx)}'].append(depth)
        else:
            # if waypoint_idx == 3 or waypoint_idx == 7:
            #     for i in range(5):
            #         data_class.images[f'{str(waypoint_idx)}'].append([np.array(rgb1), np.array(rgb2)])
            #         joint_diffs = [robot_joint_state[i]-action_prev[i]
            #                         for i in range(len(robot_joint_state))]
            #         data_class.actions[f'{str(waypoint_idx)}'].append(action)
            #         data_class.dq[f'{str(waypoint_idx)}'].append(joint_diffs)
            #         data_class.qpos[f'{str(waypoint_idx)}'].append(robot_joint_state)
            #         data_class.masks_target[f'{str(waypoint_idx)}'].append(mask_target)
            #         data_class.foveal_block[f'{str(waypoint_idx)}'].append(np.array(block))
            #         data_class.foveal_tip1[f'{str(waypoint_idx)}'].append(np.array(tip1))
            #         data_class.recorded_positions[f'{str(waypoint_idx)}'].append(recorded_positions)
            # else:
            data_class.images[f'{str(waypoint_idx)}'].append([np.array(rgb1), np.array(rgb2)])
            joint_diffs = [robot_joint_state[i]-action_prev[i]
                            for i in range(len(robot_joint_state))]
            data_class.actions[f'{str(waypoint_idx)}'].append(action)
            data_class.dq[f'{str(waypoint_idx)}'].append(joint_diffs)
            data_class.qpos[f'{str(waypoint_idx)}'].append(robot_joint_state)
            data_class.masks_no_arm[f'{str(waypoint_idx)}'].append(mask_no_arm)
            data_class.masks_target[f'{str(waypoint_idx)}'].append(mask_target)
            data_class.foveal_block[f'{str(waypoint_idx)}'].append(np.array(block))
            # data_class.foveal_tip1[f'{str(waypoint_idx)}'].append(np.array(tip1))
            # data_class.foveal_tip2[f'{str(waypoint_idx)}'].append(np.array(tip2))
            data_class.recorded_positions[f'{str(waypoint_idx)}'].append(recorded_positions)
            data_class.depths[f'{str(waypoint_idx)}'].append(depth)
        if waypoint_idx_prev != waypoint_idx:
            print(f"cnt to finish {waypoint_idx_prev} is :{cnt}")
            cnt = 0

        obs, reward, done, info = env.step(action)

        action_prev = robot_joint_state
        time_step += 1
        cnt += 1
        
        waypoint_idx_prev = waypoint_idx
        if isinstance(obs, dict) and info['is_success'] > 0 and not success and time_step < 200:
            print("Timesteps to finish:", time_step)
            success = True
            
            # Check if all waypoint counts are less than 30
            all_counts_valid = True
            for key in data_class.images.keys():
                count = len(data_class.images[key])
                print(f"Waypoint {key} has {count} samples")
                # if count >= 30:
                #     all_counts_valid = False
                #     print(f"Waypoint {key} has too many samples ({count} >= 30)")
            
            if all_counts_valid:
                # assert waypoint_idx==13
                data_class.save_data(success_cnt)
                success_cnt += 1
            else:
                print("Skipping this episode due to excessive samples")
            
        time.sleep(0.01)
    if not success and args.env == "PegBlockPickPsm2-v0":
        data_class.save_failed_data(episode_idx)
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))




if __name__ == "__main__":
    main()
