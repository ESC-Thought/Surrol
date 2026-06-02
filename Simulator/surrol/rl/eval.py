from Simulator.surrol.tasks.needle_pick_kj_edition import NeedlePick
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import gym
import time
import numpy as np
import imageio
from Simulator.surrol.const import ROOT_DIR_PATH
import Simulator.surrol.gym
import pickle
import cv2
import torch
from SKJ_SurRoL_Development_main.surrol.rl.agents.act import ACT_Policy
from typing import Optional, List, Dict
from Simulator.surrol.tasks.peg_transfer_bimanual_new_add_foveal import BiPegTransfer
# os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
def save_videos(video, dt=0.1, video_path=None, text_list=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in ['rgb1']:
                image = image_dict[cam_name]

                image = image[:, :, [2, 1, 0]] # swap B and R channel
                image = cv2.resize(image, (w // 1, h), interpolation=cv2.INTER_CUBIC)  # Upscale frame
                if text_list is not None:
                    if 'policy' in text_list[ts]:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    image = cv2.putText(image, text_list[ts], (00, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        # print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)  # Upscale frame
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')



class PolicyEvaluator:
    def __init__(self):
        self.env_name = 'BiPegTransfer-v3'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.render = False
        self.mode = 'whole'
        self.save_video = True
        self.video_dir = '/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/eval/whole_policy/0821_bipeg_act_base/'
    
        # Define model paths
        self.model_paths = {
            'whole': {
                'dir': '/media/kejianshi/E2CE2589CE2556D7/Datasets/sim_data/model/20250721_1544_bi_peg_transfer_200_rgb1',
                'ckpt': 'policy_best.ckpt'
            },
            'sub': {
                'grasp': {
                    'dir': '/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3/experiments/sub_policies/grasp/20250421_1350_bi_peg_transfer_rgb1_grasp_200',
                    'ckpt': 'policy_best.ckpt'
                },
                'handover': {
                    'dir': '/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3/experiments/sub_policies/handover/20250421_1350_bi_peg_transfer_rgb1_handover_200',
                    'ckpt': 'policy_best.ckpt'
                },
                'place': {
                    'dir': '/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3/experiments/sub_policies/place/20250421_1206_bi_peg_transfer_rgb1_place_200',
                    'ckpt': 'policy_best.ckpt'
                }
            }
        }
        
        # Create log file when initializing
        if self.save_video and not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
            
            os.makedirs(os.path.join(self.video_dir, 'success'))
            os.makedirs(os.path.join(self.video_dir, 'failed'))
        self._save_config_log()
        
        # Initialize environment and policy
        self.env = BiPegTransfer(render_mode='rgb_array',action_mode='rpy')

    def _save_config_log(self):
        """Save configuration details to a log file"""
        log_path = os.path.join(self.video_dir, 'eval_config.txt')
        config = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'env_name': self.env_name,
            'device': self.device,
            'eval_mode': self.mode,
        }
        
        with open(log_path, 'w') as f:
            f.write('Evaluation Configuration\n')
            f.write('=======================\n\n')
            f.write(f'Timestamp: {config["timestamp"]}\n')
            f.write(f'Environment: {config["env_name"]}\n')
            f.write(f'Device: {config["device"]}\n')
            f.write(f'Evaluation Mode: {config["eval_mode"]}\n\n')
            f.write('Model Paths:\n')
            
            if self.mode == 'whole':
                path = os.path.join(self.model_paths['whole']['dir'], self.model_paths['whole']['ckpt'])
                f.write(f'whole_policy: {path}\n')
            else:
                for policy_name, policy_info in self.model_paths['sub'].items():
                    path = os.path.join(policy_info['dir'], policy_info['ckpt'])
                    f.write(f'{policy_name}_policy: {path}\n')

    def load_policy(self):
        if self.mode == 'whole':
            self.act_class1 = ACT_Policy('ACT')
            path = os.path.join(self.model_paths['whole']['dir'], self.model_paths['whole']['ckpt'])
            self.act_class1.load_ckpt(self.model_paths['whole']['dir'], self.model_paths['whole']['ckpt'])

        elif self.mode == 'sub':
            self.act_class1 = ACT_Policy('ACT')
            self.act_class1.load_ckpt(
                self.model_paths['sub']['grasp']['dir'],
                self.model_paths['sub']['grasp']['ckpt']
            )

            self.act_class2 = ACT_Policy('ACT')
            self.act_class2.load_ckpt(
                self.model_paths['sub']['handover']['dir'],
                self.model_paths['sub']['handover']['ckpt']
            )

            self.act_class3 = ACT_Policy('ACT')
            self.act_class3.load_ckpt(
                self.model_paths['sub']['place']['dir'],
                self.model_paths['sub']['place']['ckpt']
            )


    def evaluate_single_episode(self, episode_num, success_cnt) -> Dict:
        """Run a single evaluation 9episode"""
        image_list = []
        obs = self.env.reset()
        episode_length = 0
        done = False
        text_list = []
        
        while not done and episode_length < 200:
            # Get action from policy
            script_action, rgb1, rgb2, mask_ori, mask_no_arm, mask_target, depth, robot_joint_state, waypoint_idx = self.env.get_oracle_action(obs)
            # if waypoint_idx in [0,1,2,3]:
            #     with torch.no_grad():
            #         print('getting actions from policy')
            #         action = self.act_class1.get_action(obs, step_idx)
            pos_obj = obs['observation'][14:17]
            
            if waypoint_idx in [0,1,2,3,4]:
                with torch.no_grad():
                    action = self.act_class1.get_action(obs, episode_length)
                obs, reward, done, info = self.env.step(action)
                text_list.append('following policy')
            # elif waypoint_idx in [9, 10, 11, 12, 13]:
            #     with torch.no_grad():
            #         print('getting actions from policy')
            #         action = self.act_class3.get_action(obs, step_idx)     
            
            # Take step in environment
            else:
                obs, reward, done, info = self.env.step(script_action)
                text_list.append('following script')

            if 'images' in obs:
                image = obs['images'] 
                image_list.append(image)
            else:
                image_list.append({'main': obs['image']})

            if pos_obj[2] > 3.6:
                done = True
            if done: 
                success_cnt += 1
            episode_length += 1

        if self.save_video:
            save_dir = os.path.join(self.video_dir, f'video{episode_num}.mp4')
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)
            save_videos(image_list, video_path=save_dir, text_list=text_list)
        return success_cnt

    def evaluate_episode(self, episode_num, success_cnt) -> Dict:
        """Run a single evaluation episode"""
        image_list = []
        obs = self.env.reset()
        step_idx = 0
        done = False
        text_list = []
        success = False
        while not success and step_idx < 200:
            # Get action from policy
            # action, rgb1, rgb2, mask_ori, mask_target, depth, robot_joint_state, waypoint_idx = self.env.get_oracle_action(obs)
            # pos_obj = obs['observation'][14:17]
            # if pos_obj[2] > 3.6:
            if step_idx < 40:
                with torch.no_grad():
                    action = self.act_class1.get_action(obs, step_idx)
                    text_list.append('following policy grasping')

            elif (step_idx >=40 and step_idx < 100):
                with torch.no_grad():
                    action = self.act_class2.get_action(obs, step_idx-40)
                    text_list.append('following policy handover')
            elif step_idx >=100:
                with torch.no_grad():
                    action = self.act_class3.get_action(obs, step_idx-100)     
                    text_list.append('following policy placing')
            # Take step in environment
            obs, reward, done, info = self.env.step(action)
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
            if info['is_success'] > 0:
                success_cnt += 1
                success=True

            step_idx += 1

        if self.save_video:
            if success:
                save_dir = os.path.join(self.video_dir, 'success', f'success_video{episode_num}.mp4')
            else:
                save_dir = os.path.join(self.video_dir, 'failed', f'failed_video{episode_num}.mp4')
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)
                os.makedirs(os.path.join(self.video_dir, 'success'))
                os.makedirs(os.path.join(self.video_dir, 'failed'))
            save_videos(image_list, video_path=save_dir, text_list=None)
            
        return success_cnt

    
    def evaluate_whole_policy(self, episode_num, success_cnt = 0):
        image_list = []
        obs = self.env.reset()
        episode_length = 0
        success = False
        while not success and episode_length < 200:
            # Get action from policy
            
            with torch.no_grad():
                action = self.act_class1.get_action(obs, episode_length)
            
            # Take step in environment
            obs, reward, done, info = self.env.step(action)
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
            episode_length += 1
            # if pos_obj[2] > 3.6:
            if info['is_success'] > 0:
                success_cnt += 1
                success=True
        if self.save_video:
            if success:
                save_dir = os.path.join(self.video_dir, 'success', f'success_video{episode_num}.mp4')
            else:
                save_dir = os.path.join(self.video_dir, 'failed', f'failed_video{episode_num}.mp4')
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)
                os.makedirs(os.path.join(self.video_dir, 'success'))
                os.makedirs(os.path.join(self.video_dir, 'failed'))
            save_videos(image_list, video_path=save_dir, text_list=None)
            
        return success_cnt
    
    def close(self):
        """Clean up resources"""
        self.env.close()

def main():
    # Create evaluator
    evaluator = PolicyEvaluator()
    evaluator.load_policy()
    if evaluator.mode == 'whole':
        total_trials = 100
        success_cnt =0
        for i in range(total_trials):
            success_cnt = evaluator.evaluate_whole_policy(i, success_cnt=success_cnt)
            print(f'Eval tial number: {i+1}; Success rate: {success_cnt}/{i+1}')
    else:
        success_cnt = 0
        for i in range(100):
            success_cnt = evaluator.evaluate_episode(i, success_cnt)
            print(f'Eval tial number: {i+1}; Success rate: {success_cnt}/{i+1}')
if __name__ == "__main__":

    print('start')
    main()
