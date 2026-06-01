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
parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
args = parser.parse_args()

actions_list = []
image_list = []
qpos_list = []

images = []  # record video
mask_list = []
depth_list = []

def main():
    env = gym.make(args.env, render_mode='human')  # 'human'
    num_itr = 1000 if not args.video else 1 # TODO
    cnt = 0
    init_state_space = 'random'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env._max_episode_steps

    print()
    while len(actions_list) < num_itr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions_list))
        goToGoal(env, obs)
        cnt += 1
    
    save_data()

    used_time = time.time() - init_time
    # print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {cnt}")
    env.close()


def goToGoal(env, last_obs):


    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()

    obs, success = last_obs, False

    while time_step < min(env._max_episode_steps, args.steps) and not success:
        action, rgb1, mask, depth, robot_joint_state = env.get_oracle_action(obs)
        image_list.append(rgb1)
        actions_list.append(action)
        qpos_list.append(robot_joint_state)
        mask_list.append(mask)
        depth_list.append(depth)

        if args.video:
            # img, mask = env.render('img_array')
            img = env.render('rgb_array')
            images.append(img)
            # masks.append(mask)

        obs, reward, done, info = env.step(action)

        time_step += 1

        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True

        time.sleep(0.01)
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))


def save_data():
    """
    Save the collected data to a pickle file.
    """
    data = {
        'images': image_list,
        'actions': actions_list,  # TODO 
        # 'actions': self.new_actions,
        'qpos': qpos_list,
        'masks': mask_list,
        'depths': depth_list
    }
    # save_path = f'/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/image_action_qpos_{self.episode_idx}.pkl'
    save_path = f'/root/autodl-tmp/IROS_SurRoL/rl/act-main-3/data/1222exp3-1/image_action_qpos.pkl'

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved data to {save_path}")



if __name__ == "__main__":
    main()
