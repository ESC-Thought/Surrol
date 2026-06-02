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
from surrol.tasks.peg_transfer_bimanual_new import BiPegTransfer
# Remove pynput import
# from pynput import keyboard as pynput_keyboard
import threading
import tkinter as tk
from functools import partial
# os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
import pybullet as p
import matplotlib.pyplot as plt
import h5py
import math
from haptic_src.touch_haptic import (
    initTouch_right, closeTouch_right, getDeviceAction_right, 
    initTouch_left, closeTouch_left, getDeviceAction_left,
    startScheduler, stopScheduler
)

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
        self.env_name = 'BiPegTransfer-v2'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.render = False
        self.mode = 'whole'
        self.save_video = True
        self.video_dir = '/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/eval/whole_policy/0428-stereo-100_correction/'
        self.wait_for_user_input = True
        self.keyboard_control = True  # New flag for keyboard control
        self.control_step_size = 1  # Step size for keyboard control movements
        
        # Define model paths
        self.model_paths = {
            'whole': {
                'dir': '/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/ACTBasePolicy_2',
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
        self.env = BiPegTransfer(render_mode='human', action_mode='yaw')

        # Add haptic device support
        self.use_haptic = True
        self.haptic_initialized = False
        self.haptic_scale = 0.03  # Reduced scale factor for haptic input (from 0.9 to 0.3)
        
    def _save_config_log(self):
        """Save configuration details to a log file"""
        log_path = os.path.join(self.video_dir, 'eval_config.txt')
        config = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'env_name': self.env_name,
            'device': self.device,
            'eval_mode': self.mode,
            'wait_for_user_input': self.wait_for_user_input,
            'keyboard_control': self.keyboard_control,
        }
        
        with open(log_path, 'w') as f:
            f.write('Evaluation Configuration\n')
            f.write('=======================\n\n')
            f.write(f'Timestamp: {config["timestamp"]}\n')
            f.write(f'Environment: {config["env_name"]}\n')
            f.write(f'Device: {config["device"]}\n')
            f.write(f'Evaluation Mode: {config["eval_mode"]}\n')
            f.write(f'Wait for User Input: {config["wait_for_user_input"]}\n')
            f.write(f'Keyboard Control: {config["keyboard_control"]}\n\n')
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

    def wait_for_user_command(self):
        """Wait for user to press Enter before continuing"""
        if self.wait_for_user_input:
            input("Press Enter to apply the next action...")

    def initialize_haptic_devices(self):
        """Initialize the haptic devices for robot control"""
        if not self.haptic_initialized:
            try:
                print("Initializing haptic devices...")
                initTouch_right()
                initTouch_left()
                startScheduler()
                self.haptic_initialized = True
                print("Haptic devices initialized successfully")
                return True
            except Exception as e:
                print(f"Failed to initialize haptic devices: {e}")
                return False
        return True
        
    def cleanup_haptic_devices(self):
        """Clean up haptic device resources"""
        if self.haptic_initialized:
            try:
                stopScheduler()
                closeTouch_right()
                closeTouch_left()
                self.haptic_initialized = False
                print("Haptic devices closed")
            except Exception as e:
                print(f"Error closing haptic devices: {e}")

    def gui_control_robot(self, action, accumulated_corrections=None, episode_num=None):
        try:
            # Initialize haptic devices if using them
            if self.use_haptic and not self.initialize_haptic_devices():
                print("Falling back to GUI control due to haptic device initialization failure")
                self.use_haptic = False
            
            # Create the main window for GUI control
            root = tk.Tk()
            root.title("Robot Control Interface")
            root.geometry("800x600")
            
            # Set up fonts
            title_font = ('Helvetica', 16, 'bold')
            label_font = ('Helvetica', 12)
            button_font = ('Helvetica', 10)
            
            # Create control data dictionary
            control_data = {
                'action': action.copy(),
                'done': False,
                'exit': False,
                'reset': False,
                'quit': False,
                'step_size': 0.05,
                'accumulated_corrections': accumulated_corrections,
                'last_update_time': time.time(),
                'episode_num': episode_num,
                'control_mode': 'position'  # Start in position mode
            }
            
            # Function to apply action and update environment
            def apply_action():
                current_time = time.time()
                # Throttle updates to prevent excessive environment steps
                if current_time - control_data['last_update_time'] < 0.1:  # 100ms minimum between actions
                    return
                
                # Convert RPY action to yaw mode before applying
                action_yaw = np.zeros(10)
                # Right arm: xyz, yaw, gripper (indices 0-4)
                action_yaw[0:3] = control_data['action'][0:3]  # xyz position
                action_yaw[3] = control_data['action'][5]      # yaw
                action_yaw[4] = control_data['action'][6]      # gripper
                # Left arm: xyz, yaw, gripper (indices 5-9)
                action_yaw[5:8] = control_data['action'][7:10]  # xyz position
                action_yaw[8] = control_data['action'][12]      # yaw
                action_yaw[9] = control_data['action'][13]      # gripper
                
                # Apply the action to the environment
                obs, reward, done, info = self.env.step(action_yaw)
                
                # Update the display
                update_action_display()
                
                # Record the time of the update
                control_data['last_update_time'] = current_time
            
            # Function to reset episode
            def reset_episode():
                print("\n=== RESET REQUESTED BY USER ===")
                # Reset the environment immediately
                self.env.reset()
                print("=== Environment reset complete ===")
                
                # Set flags to exit the GUI
                control_data['done'] = True
                control_data['exit'] = True
                control_data['reset'] = True
                root.destroy()
            
            # Function to exit control mode
            def quit_control():
                control_data['done'] = True
                control_data['exit'] = True
                control_data['quit'] = True
                # Clean up haptic devices before destroying window
                if self.use_haptic:
                    self.cleanup_haptic_devices()
                root.destroy()
            
            # Function to update history
            def update_history():
                # Add the current action to the accumulated corrections
                control_data['accumulated_corrections'].append(control_data['action'].copy())
            
            # Create main frame
            main_frame = tk.Frame(root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Create title
            title_label = tk.Label(main_frame, text="Robot Control Interface", font=title_font)
            title_label.pack(pady=10)
            
            # Create action display frame
            action_frame = tk.Frame(main_frame)
            action_frame.pack(fill=tk.X, pady=10)
            
            # Display current action values
            action_label = tk.Label(action_frame, text="Current Action:", font=label_font)
            action_label.pack(anchor=tk.W)
            
            # Create a frame for the action values
            values_frame = tk.Frame(action_frame)
            values_frame.pack(fill=tk.X, pady=5)
            
            # Right arm position and orientation
            right_arm_frame = tk.Frame(values_frame)
            right_arm_frame.pack(side=tk.LEFT, padx=10)
            
            tk.Label(right_arm_frame, text="Right Arm", font=label_font).pack()
            
            right_pos_frame = tk.Frame(right_arm_frame)
            right_pos_frame.pack(fill=tk.X)
            
            tk.Label(right_pos_frame, text="Position:").pack(side=tk.LEFT)
            right_pos_label = tk.Label(right_pos_frame, text=f"X: {action[0]:.3f}, Y: {action[1]:.3f}, Z: {action[2]:.3f}")
            right_pos_label.pack(side=tk.LEFT, padx=5)
            
            right_orn_frame = tk.Frame(right_arm_frame)
            right_orn_frame.pack(fill=tk.X)
            
            tk.Label(right_orn_frame, text="Orientation:").pack(side=tk.LEFT)
            right_orn_label = tk.Label(right_orn_frame, text=f"R: {action[3]:.3f}, P: {action[4]:.3f}, Y: {action[5]:.3f}")
            right_orn_label.pack(side=tk.LEFT, padx=5)
            
            right_grip_frame = tk.Frame(right_arm_frame)
            right_grip_frame.pack(fill=tk.X)
            
            tk.Label(right_grip_frame, text="Gripper:").pack(side=tk.LEFT)
            right_grip_label = tk.Label(right_grip_frame, text=f"{action[6]:.3f}")
            right_grip_label.pack(side=tk.LEFT, padx=5)
            
            # Left arm position and orientation
            left_arm_frame = tk.Frame(values_frame)
            left_arm_frame.pack(side=tk.LEFT, padx=10)
            
            tk.Label(left_arm_frame, text="Left Arm", font=label_font).pack()
            
            left_pos_frame = tk.Frame(left_arm_frame)
            left_pos_frame.pack(fill=tk.X)
            
            tk.Label(left_pos_frame, text="Position:").pack(side=tk.LEFT)
            left_pos_label = tk.Label(left_pos_frame, text=f"X: {action[7]:.3f}, Y: {action[8]:.3f}, Z: {action[9]:.3f}")
            left_pos_label.pack(side=tk.LEFT, padx=5)
            
            left_orn_frame = tk.Frame(left_arm_frame)
            left_orn_frame.pack(fill=tk.X)
            
            tk.Label(left_orn_frame, text="Orientation:").pack(side=tk.LEFT)
            left_orn_label = tk.Label(left_orn_frame, text=f"R: {action[10]:.3f}, P: {action[11]:.3f}, Y: {action[12]:.3f}")
            left_orn_label.pack(side=tk.LEFT, padx=5)
            
            left_grip_frame = tk.Frame(left_arm_frame)
            left_grip_frame.pack(fill=tk.X)
            
            tk.Label(left_grip_frame, text="Gripper:").pack(side=tk.LEFT)
            left_grip_label = tk.Label(left_grip_frame, text=f"{action[13]:.3f}")
            left_grip_label.pack(side=tk.LEFT, padx=5)
            
            # Function to update action display
            def update_action_display():
                right_pos_label.config(text=f"X: {control_data['action'][0]:.3f}, Y: {control_data['action'][1]:.3f}, Z: {control_data['action'][2]:.3f}")
                right_orn_label.config(text=f"R: {control_data['action'][3]:.3f}, P: {control_data['action'][4]:.3f}, Y: {control_data['action'][5]:.3f}")
                right_grip_label.config(text=f"{control_data['action'][6]:.3f}")
                
                left_pos_label.config(text=f"X: {control_data['action'][7]:.3f}, Y: {control_data['action'][8]:.3f}, Z: {control_data['action'][9]:.3f}")
                left_orn_label.config(text=f"R: {control_data['action'][10]:.3f}, P: {control_data['action'][11]:.3f}, Y: {control_data['action'][12]:.3f}")
                left_grip_label.config(text=f"{control_data['action'][13]:.3f}")
            
            # Create control panel
            control_panel = tk.Frame(main_frame)
            control_panel.pack(fill=tk.X, pady=10)
            
            # Step size controls
            step_frame = tk.Frame(control_panel)
            step_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(step_frame, text="Step Size:", font=label_font).pack(side=tk.LEFT)
            
            def decrease_step():
                control_data['step_size'] = max(0.01, control_data['step_size'] - 0.01)
                step_size_label.config(text=f"{control_data['step_size']:.2f}")
                status_label.config(text=f"Step size decreased to {control_data['step_size']:.2f}")
            
            def increase_step():
                control_data['step_size'] = min(1.0, control_data['step_size'] + 0.01)
                step_size_label.config(text=f"{control_data['step_size']:.2f}")
                status_label.config(text=f"Step size increased to {control_data['step_size']:.2f}")
            
            step_decrease_button = tk.Button(step_frame, text="-", command=decrease_step, width=3, height=1, font=button_font)
            step_decrease_button.pack(side=tk.LEFT, padx=5)
            
            step_size_label = tk.Label(step_frame, text=f"{control_data['step_size']:.2f}", width=5)
            step_size_label.pack(side=tk.LEFT, padx=5)
            
            step_increase_button = tk.Button(step_frame, text="+", command=increase_step, width=3, height=1, font=button_font)
            step_increase_button.pack(side=tk.LEFT, padx=5)
            
            # Reset button
            reset_button = tk.Button(step_frame, text="Reset Episode", command=reset_episode, width=15, height=1, font=button_font, bg="#ffaa66")
            reset_button.pack(side=tk.RIGHT, padx=10)
            
            # Quit button
            quit_button = tk.Button(step_frame, text="Quit", command=quit_control, width=10, height=1, font=button_font, bg="#ff6666")
            quit_button.pack(side=tk.RIGHT, padx=10)
            
            # Add a mode toggle for position vs. orientation control
            def toggle_control_mode():
                if control_data['control_mode'] == 'position':
                    control_data['control_mode'] = 'orientation'
                    mode_button.config(text="Mode: ORIENTATION", bg="#ffaa66")
                    status_label.config(text="Switched to ORIENTATION control mode")
                else:
                    control_data['control_mode'] = 'position'
                    mode_button.config(text="Mode: POSITION", bg="#66aaff")
                    status_label.config(text="Switched to POSITION control mode")
            
            # Add mode button to control panel
            mode_button = tk.Button(
                step_frame,
                text="Mode: POSITION",
                command=toggle_control_mode,
                width=15,
                height=1,
                font=button_font,
                bg="#66aaff"
            )
            mode_button.pack(side=tk.RIGHT, padx=10)
            
            # Add haptic toggle button
            def toggle_haptic_control():
                self.use_haptic = not self.use_haptic
                
                if self.use_haptic:
                    if self.initialize_haptic_devices():
                        haptic_button.config(text="Haptic: ON", bg="#66ff66")
                        status_label.config(text="Switched to haptic device control")
                        # Start haptic control loop
                        root.after(50, haptic_control_loop)
                    else:
                        self.use_haptic = False
                        haptic_button.config(text="Haptic: OFF", bg="#ff6666")
                        status_label.config(text="Failed to initialize haptic devices")
                else:
                    haptic_button.config(text="Haptic: OFF", bg="#ff6666")
                    status_label.config(text="Switched to GUI control")
            
            haptic_button = tk.Button(
                step_frame,
                text="Haptic: OFF" if not self.use_haptic else "Haptic: ON",
                command=toggle_haptic_control,
                width=15,
                height=1,
                font=button_font,
                bg="#ff6666" if not self.use_haptic else "#66ff66"
            )
            haptic_button.pack(side=tk.LEFT, padx=10)
            
            # Status display
            status_frame = tk.Frame(main_frame)
            status_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(status_frame, text="Status:", font=label_font).pack(side=tk.LEFT)
            status_label = tk.Label(status_frame, text="Ready")
            status_label.pack(side=tk.LEFT, padx=5)
            
            # Create a frame for the gripper controls
            gripper_frame = tk.Frame(main_frame)
            gripper_frame.pack(fill=tk.X, pady=10)
            
            # Right gripper controls
            right_gripper_frame = tk.Frame(gripper_frame)
            right_gripper_frame.pack(side=tk.LEFT, padx=20)
            
            tk.Label(right_gripper_frame, text="Right Gripper", font=label_font).pack()
            
            def open_right_gripper():
                control_data['action'][6] = 1.0
                status_label.config(text="Opened right gripper")
                apply_action()
            
            def close_right_gripper():
                control_data['action'][6] = -0.5
                status_label.config(text="Closed right gripper")
                apply_action()
            
            open_right_button = tk.Button(right_gripper_frame, text="Open", command=open_right_gripper, width=10, height=2, font=button_font)
            open_right_button.pack(side=tk.LEFT, padx=5)
            
            close_right_button = tk.Button(right_gripper_frame, text="Close", command=close_right_gripper, width=10, height=2, font=button_font)
            close_right_button.pack(side=tk.LEFT, padx=5)
            
            # Left gripper controls
            left_gripper_frame = tk.Frame(gripper_frame)
            left_gripper_frame.pack(side=tk.LEFT, padx=20)
            
            tk.Label(left_gripper_frame, text="Left Gripper", font=label_font).pack()
            
            def open_left_gripper():
                control_data['action'][13] = 1.0
                status_label.config(text="Opened left gripper")
                apply_action()
            
            def close_left_gripper():
                control_data['action'][13] = -0.5
                status_label.config(text="Closed left gripper")
                apply_action()
            
            open_left_button = tk.Button(left_gripper_frame, text="Open", command=open_left_gripper, width=10, height=2, font=button_font)
            open_left_button.pack(side=tk.LEFT, padx=5)
            
            close_left_button = tk.Button(left_gripper_frame, text="Close", command=close_left_gripper, width=10, height=2, font=button_font)
            close_left_button.pack(side=tk.LEFT, padx=5)
            
            # Create a frame for the done button
            done_frame = tk.Frame(main_frame)
            done_frame.pack(fill=tk.X, pady=20)
            
            def finish_control():
                control_data['done'] = True
                # Clean up haptic devices before destroying window
                if self.use_haptic:
                    self.cleanup_haptic_devices()
                root.destroy()
            
            done_button = tk.Button(done_frame, text="Done", command=finish_control, width=20, height=2, font=button_font, bg="#66ff66")
            done_button.pack()
            
            # Haptic device control loop for yaw mode
            def haptic_control_loop():
                if self.use_haptic:
                    # Get haptic device input for right device
                    right_action = np.array([0, 0, 0, 0, 0], dtype=np.float32)
                    getDeviceAction_right(right_action)
                    
                    # Get haptic device input for left device
                    left_action = np.array([0, 0, 0, 0, 0], dtype=np.float32)
                    getDeviceAction_left(left_action)
                    
                    # Initialize yaw mode action array [xyz_right, yaw_right, grip_right, xyz_left, yaw_left, grip_left]
                    action_yaw = np.zeros(10)
                    
                    # Check for quit or reset conditions
                    if control_data['quit'] or control_data['reset']:
                        # Clean up haptic devices
                        if self.use_haptic:
                            self.cleanup_haptic_devices()
                        # Return appropriate command
                        if control_data['reset']:
                            return "RESET"
                        if control_data['quit']:
                            return "QUIT"
                    
                    # Process right device input
                    if right_action[4] == 2:  # Clutch mode
                        status_label.config(text="Right clutch engaged")
                        action_yaw[0:4] = 0  # Zero out xyz and yaw
                    else:
                        # Map position (touch device -> robot)
                        action_yaw[0] = right_action[2] * self.haptic_scale  # Z -> X
                        action_yaw[1] = right_action[0] * self.haptic_scale  # X -> Y
                        action_yaw[2] = right_action[1] * self.haptic_scale  # Y -> Z
                        
                        # Map yaw rotation
                        yaw_input = -right_action[3]/math.pi*180*0.1
                        if abs(yaw_input) > 0.05:  # Deadzone
                            action_yaw[3] = yaw_input  # Yaw control
                        
                        # Map gripper state
                        if right_action[4] == 0:  # Open
                            action_yaw[4] = 1.0
                            status_label.config(text="Right gripper opened")
                        elif right_action[4] == 1:  # Close
                            action_yaw[4] = -0.5
                            status_label.config(text="Right gripper closed")
                    
                    # Process left device input
                    if left_action[4] == 2:  # Clutch mode
                        status_label.config(text="Left clutch engaged")
                        action_yaw[5:9] = 0  # Zero out xyz and yaw
                    else:
                        # Map position (touch device -> robot)
                        action_yaw[5] = left_action[2] * self.haptic_scale  # Z -> X
                        action_yaw[6] = left_action[0] * self.haptic_scale  # X -> Y
                        action_yaw[7] = left_action[1] * self.haptic_scale  # Y -> Z
                        
                        # Map yaw rotation
                        yaw_input = -left_action[3]/math.pi*180*0.1
                        if abs(yaw_input) > 0.05:  # Deadzone
                            action_yaw[8] = yaw_input  # Yaw control
                        
                        # Map gripper state
                        if left_action[4] == 0:  # Open
                            action_yaw[9] = 1.0
                            status_label.config(text="Left gripper opened")
                        elif left_action[4] == 1:  # Close
                            action_yaw[9] = -0.5
                            status_label.config(text="Left gripper closed")
                    
                    # Apply the yaw mode action directly to the environment
                    obs, reward, done, info = self.env.step(action_yaw)
                    
                    # Schedule the next update if still using haptic and not quitting
                    if not control_data['exit'] and self.use_haptic:
                        root.after(50, haptic_control_loop)  # 20 Hz update rate
            
            # Start haptic control loop if enabled
            if self.use_haptic:
                root.after(50, haptic_control_loop)
            
            # Run the GUI
            root.mainloop()
            
            # Handle exit conditions
            if control_data['reset']:
                return "RESET"
            if control_data['quit']:
                return "QUIT"
            
            return control_data['action']
            
        except Exception as e:
            print(f"Error in GUI control: {str(e)}")
            if self.use_haptic:
                self.cleanup_haptic_devices()
            return "QUIT"

    def evaluate_whole_policy(self, episode_num, success_cnt=0):
        """Evaluate the whole policy with data collection"""
        print(f"\nStarting evaluation episode {episode_num}")
        
        try:
            # Use collect_correction_data instead of manual evaluation
            success, episode_length = self.collect_correction_data(episode_num)
            
            # Update success count
            if success:
                success_cnt += 1
                print(f"Episode {episode_num} succeeded in {episode_length} steps")
                if self.save_video:
                    # Move successful episode data to success directory
                    src_dir = os.path.join(self.video_dir, f'episode_{episode_num:03d}')
                    dst_dir = os.path.join(self.video_dir, 'success', f'episode_{episode_num:03d}')
                    if os.path.exists(src_dir):
                        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                        os.rename(src_dir, dst_dir)
            else:
                print(f"Episode {episode_num} failed after {episode_length} steps")
                if self.save_video:
                    # Move failed episode data to failed directory
                    src_dir = os.path.join(self.video_dir, f'episode_{episode_num:03d}')
                    dst_dir = os.path.join(self.video_dir, 'failed', f'episode_{episode_num:03d}')
                    if os.path.exists(src_dir):
                        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                        os.rename(src_dir, dst_dir)
            
            return success_cnt
        
        except Exception as e:
            print(f"Error in episode {episode_num}: {str(e)}")
            import traceback
            traceback.print_exc()
            return success_cnt

    def close(self):
        """Clean up resources"""
        self.env.close()

    def save_corrections(self, corrections, episode_num=None):
        """Save correction data in HDF5 format"""
        import h5py
        
        # Create directory for correction data if it doesn't exist
        corrections_dir = os.path.join(self.video_dir, 'corrections')
        if not os.path.exists(corrections_dir):
            os.makedirs(corrections_dir)
        
        # Generate filename with timestamp and episode number if provided
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if episode_num is not None:
            filename = f'corrections_{timestamp}_episode_{episode_num:03d}.h5'
        else:
            filename = f'corrections_{timestamp}.h5'
        
        filepath = os.path.join(corrections_dir, filename)
        
        # Save corrections to HDF5 file
        with h5py.File(filepath, 'w') as hf:
            # Store metadata
            hf.attrs['timestamp'] = timestamp
            hf.attrs['num_corrections'] = len(corrections)
            if episode_num is not None:
                hf.attrs['episode_num'] = episode_num
            
            # Create a group for all corrections
            corrections_group = hf.create_group('corrections')
            
            # Store each correction
            for i, correction in enumerate(corrections):
                # Create a group for this correction
                corr_group = corrections_group.create_group(f'correction_{i:03d}')
                
                # Store all attributes of the correction
                for key, value in correction.items():
                    if isinstance(value, (int, float, str, bool)):
                        corr_group.attrs[key] = value
                    elif hasattr(value, 'shape'):  # For numpy arrays
                        corr_group.create_dataset(key, data=value)
                    elif isinstance(value, dict):  # For nested dictionaries
                        nested_group = corr_group.create_group(key)
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, (int, float, str, bool)):
                                nested_group.attrs[nested_key] = nested_value
                            elif hasattr(nested_value, 'shape'):
                                nested_group.create_dataset(nested_key, data=nested_value)
        
        print(f"Saved {len(corrections)} corrections to {filepath}")
        return filepath

    def verify_action_format(self, action):
        """Verify and normalize the action format to ensure it's compatible with the environment"""
        # Make a copy to avoid modifying the original
        normalized_action = action.copy()
        
        # Ensure gripper values are exactly 0.0 or 1.0
        # normalized_action[6] = 1.0 if normalized_action[6] > 0 else 0.0  # Right gripper
        # normalized_action[13] = 1.0 if normalized_action[13] > 0 else 0.0  # Left gripper
        
        # Print debug information
        if not np.array_equal(normalized_action, action):
            print(f"Action normalized:")
            print(f"  Original: {action}")
            print(f"  Normalized: {normalized_action}")
        
        return normalized_action

    def open_multiple_views(self):
        """Open multiple windows with different camera views and modalities"""
        try:
            # Get the scaling from the environment
            scaling = self.env.SCALING if hasattr(self.env, 'SCALING') else 5.0
            
            # Create a new function to capture and display camera views
            def create_view_windows(view_type):
                # Set up the camera based on view type
                if view_type == "top":
                    # Top-down view
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=0.6 * scaling,
                        cameraYaw=0,
                        cameraPitch=-60,
                        cameraTargetPosition=(0.55, 0, 0.686)
                    )
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                    title_prefix = "Top-Down"
                elif view_type == "front":
                    # Front view
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=0.9 * scaling,
                        cameraYaw=90,
                        cameraPitch=-30,
                        cameraTargetPosition=(0.55, 0, 0.7)
                    )
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                    title_prefix = "Front"
                elif view_type == "side":
                    # Side view
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=0.9 * scaling,
                        cameraYaw=0,
                        cameraPitch=-30,
                        cameraTargetPosition=(0.55, 0, 0.7)
                    )
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                    title_prefix = "Side"
                
                # Capture the current view with all modalities
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                    width=640,
                    height=480,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL
                )
                
                # Process and display RGB view
                rgb_array = np.array(rgbImg)
                rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
                rgb_title = f"{title_prefix} RGB"
                cv2.imshow(rgb_title, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
                
                # Process and display depth view
                depth_array = np.array(depthImg)
                # Normalize depth for visualization
                depth_norm = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array) + 1e-6)
                depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                depth_title = f"{title_prefix} Depth"
                cv2.imshow(depth_title, depth_colormap)
                
                # Process and display segmentation view
                seg_array = np.array(segImg)
                # Create a colorful visualization of segmentation
                seg_colormap = np.zeros((height, width, 3), dtype=np.uint8)
                unique_ids = np.unique(seg_array)
                for id_val in unique_ids:
                    if id_val == 0:  # Background
                        continue
                    # Generate a unique color for each object ID
                    color = np.array([
                        (id_val * 95) % 255,
                        (id_val * 130) % 255,
                        (id_val * 65) % 255
                    ], dtype=np.uint8)
                    seg_colormap[seg_array == id_val] = color
                seg_title = f"{title_prefix} Segmentation"
                cv2.imshow(seg_title, seg_colormap)
                
                # Update all windows
                cv2.waitKey(1)
                
                return [rgb_title, depth_title, seg_title]
            
            # Create all views
            view_titles = []
            for view in ["top", "front", "side"]:
                titles = create_view_windows(view)
                view_titles.extend(titles)
            
            print(f"Opened {len(view_titles)} view windows")
            
            # Schedule periodic updates of the views
            def update_views():
                if hasattr(self, 'multi_view_active') and self.multi_view_active:
                    for view in ["top", "front", "side"]:
                        create_view_windows(view)
                    # Schedule next update
                    cv2.waitKey(1)
                    return True
                return False
            
            # Start the update loop in a separate thread
            self.multi_view_active = True
            
            def view_update_thread():
                while update_views():
                    time.sleep(0.5)  # Update every 500ms
            
            threading.Thread(target=view_update_thread, daemon=True).start()
            
            return "Multiple views opened successfully"
            
        except Exception as e:
            print(f"Error creating multiple views: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def collect_correction_data(self, episode_num):
        """Collect data for robot corrections with comprehensive logging in HDF5 format"""
        while True:  # Loop to allow episode resets
            try:
                # Initialize episode
                print("\n=== Starting new episode - resetting environment ===")
                obs = self.env.reset()
                print("=== Environment reset complete ===\n")
                
                episode_length = 0
                success = False
                act_policy = []
                correction_indicator = []
                act_corrected = []
                obs_list = []
                state_before_corrections = []
                state_after_corrections = []
                reset_requested = False
                
                while not success and episode_length < 200:
                    # Get policy action prediction
                    with torch.no_grad():
                        policy_action = self.act_class1.get_action(obs, episode_length)
                    act_policy.append(policy_action)
                    obs_list.append(obs)
                    
                    # Ask for correction with error handling
                    try:
                        print("\nCurrent robot pose applied. Do you want to modify it?")
                        print("Press 'm' to modify, any other key to continue")
                        modify = input().lower()  # Move input() after the prompts
                    except (OSError, IOError) as e:
                        print(f"Input error occurred: {e}")
                        print("Exiting episode...")
                        return False, episode_length
                    
                    # Initialize correction vector
                    correction1 = np.zeros(7)
                    correction2 = np.zeros(7)
                    needs_correction = modify == 'm'
                    
                    if needs_correction:
                        try:
                            # Ask for control method
                            print("Use haptic device? (y/n): ")
                            control_method = input().lower()  # Move input() after the prompt
                            self.use_haptic = control_method == 'y'
                        except (OSError, IOError) as e:
                            print(f"Input error occurred: {e}")
                            print("Defaulting to GUI control")
                            self.use_haptic = False
                        
                        # Store robot state before correction
                        state_before = self.get_robot_state(obs)
                        
                        # Call the GUI control function
                        control_action = self.gui_control_robot(policy_action, accumulated_corrections=[], episode_num=episode_num)
                        
                        # Check if control_action is a special command
                        if isinstance(control_action, str):
                            if control_action == "RESET":
                                print("\n=== Reset detected, starting new episode ===")
                                reset_requested = True
                                break
                            
                            if control_action == "QUIT":
                                return False, episode_length
                        else:
                            # It's a normal action, process it
                            correction1[:6] += control_action[:6]
                            correction2[:6] += control_action[7:13]
                            final_pose = control_action.copy()
                            
                            # Convert RPY action to yaw mode
                            action_yaw = np.zeros(10)
                            # Right arm: xyz, yaw, gripper (indices 0-4)
                            action_yaw[0:3] = control_action[0:3]  # xyz position
                            action_yaw[3] = control_action[5]      # yaw
                            action_yaw[4] = control_action[6]      # gripper
                            # Left arm: xyz, yaw, gripper (indices 5-9)
                            action_yaw[5:8] = control_action[7:10]  # xyz position
                            action_yaw[8] = control_action[12]      # yaw
                            action_yaw[9] = control_action[13]      # gripper
                            print(action_yaw)
                            obs, reward, done, info = self.env.step(action_yaw)
                        
                        if reset_requested:
                            break
                        
                        # Store robot state after correction
                        state_after = self.get_robot_state(obs)
                        
                        # Calculate actual correction as the difference between states
                        actual_correction = self.calculate_state_difference(state_before, state_after)
                        
                        if sum(correction1[:6]) != 0 or sum(correction2[:6]) != 0:
                            correction_indicator.append(1)
                            state_before_corrections.append(state_before)
                            state_after_corrections.append(state_after)
                        else:
                            correction_indicator.append(0)
                            state_before_corrections.append(None)
                            state_after_corrections.append(None)
                        
                        # Store the actual correction instead of the input correction
                        act_corrected.append(actual_correction)
                    else:
                        # No correction needed, convert policy action to yaw mode
                        action_yaw = np.zeros(10)
                        # Right arm: xyz, yaw, gripper (indices 0-4)
                        action_yaw[0:3] = policy_action[0:3]  # xyz position
                        action_yaw[3] = policy_action[5]      # yaw
                        action_yaw[4] = policy_action[6]      # gripper
                        # Left arm: xyz, yaw, gripper (indices 5-9)
                        action_yaw[5:8] = policy_action[7:10]  # xyz position
                        action_yaw[8] = policy_action[12]      # yaw
                        action_yaw[9] = policy_action[13]      # gripper
                        
                        obs, reward, done, info = self.env.step(action_yaw)
                        correction_indicator.append(0)
                        act_corrected.append(np.zeros(14))  # No correction
                        state_before_corrections.append(None)
                        state_after_corrections.append(None)
                    
                    # Check for success
                    if 'is_success' in info and info['is_success'] > 0:
                        success = True
                        print("Success!")
                    
                    episode_length += 1
                
                # If reset was requested, start a new episode
                if reset_requested:
                    continue
                
                # Save data if successful
                if success:
                    # Create directory for this episode
                    episode_dir = os.path.join(self.video_dir, 'success', f'episode_{episode_num}')
                    os.makedirs(episode_dir, exist_ok=True)
                    
                    # Save data to HDF5 file
                    h5_path = os.path.join(episode_dir, f'episode_{episode_num}_data.h5')
                    
                    # Save data to HDF5 file
                    with h5py.File(h5_path, 'w') as f:
                        # Create groups
                        obs_grp = f.create_group('observations')
                        act_policy_grp = f.create_group('actions_policy')
                        correction_grp = f.create_group('corrections')
                        
                        # Save observations
                        for i, obs_dict in enumerate(obs_list):
                            obs_step = obs_grp.create_group(f'step_{i}')
                            for key, value in obs_dict.items():
                                if isinstance(value, np.ndarray):
                                    obs_step.create_dataset(key, data=value)
                        
                        # Save policy actions
                        for i, act in enumerate(act_policy):
                            act_policy_grp.create_dataset(f'step_{i}', data=act)
                        
                        # Save correction data
                        correction_grp.create_dataset('indicator', data=np.array(correction_indicator))
                        correction_grp.create_dataset('action', data=np.array(act_corrected))
                        
                        # Save state differences
                        state_diff_grp = correction_grp.create_group('state_differences')
                        for i, (indicator, before, after) in enumerate(zip(correction_indicator, state_before_corrections, state_after_corrections)):
                            if indicator > 0 and before is not None and after is not None:
                                step_grp = state_diff_grp.create_group(f'step_{i}')
                                
                                # Save task space differences
                                if 'ee_pos_right' in before and 'ee_pos_right' in after:
                                    step_grp.create_dataset('ee_pos_right_diff', data=after['ee_pos_right'] - before['ee_pos_right'])
                                if 'ee_orn_right' in before and 'ee_orn_right' in after:
                                    step_grp.create_dataset('ee_orn_right_diff', data=after['ee_orn_right'] - before['ee_orn_right'])
                                if 'ee_pos_left' in before and 'ee_pos_left' in after:
                                    step_grp.create_dataset('ee_pos_left_diff', data=after['ee_pos_left'] - before['ee_pos_left'])
                                if 'ee_orn_left' in before and 'ee_orn_left' in after:
                                    step_grp.create_dataset('ee_orn_left_diff', data=after['ee_orn_left'] - before['ee_orn_left'])
                                
                                # Save joint space differences
                                if 'qpos' in before and 'qpos' in after:
                                    step_grp.create_dataset('qpos_diff', data=after['qpos'] - before['qpos'])
                                    step_grp.create_dataset('qpos_before', data=before['qpos'])
                                    step_grp.create_dataset('qpos_after', data=after['qpos'])
                        
                        # Add metadata
                        f.attrs['success'] = success
                        f.attrs['episode_length'] = episode_length
                        f.attrs['timestamp'] = time.time()
                    
                    print(f"Saved successful episode data to {h5_path}")
                else:
                    print("Episode failed - no data saved")
                
                return success, episode_length
                
            except Exception as e:
                print(f"Error in episode {episode_num}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False, episode_length

    def get_robot_state(self, obs):
        """Extract robot state from observation"""
        state = {}
        
        # Extract joint positions (qpos) - this is the most accurate representation of robot state
        if 'qpos' in obs:
            state['qpos'] = obs['qpos'].copy()
        
        # Extract end-effector positions and orientations
        if 'ee_pos_right' in obs:
            state['ee_pos_right'] = obs['ee_pos_right'].copy()
        if 'ee_orn_right' in obs:
            state['ee_orn_right'] = obs['ee_orn_right'].copy()
        if 'ee_pos_left' in obs:
            state['ee_pos_left'] = obs['ee_pos_left'].copy()
        if 'ee_orn_left' in obs:
            state['ee_orn_left'] = obs['ee_orn_left'].copy()
        
        # Extract gripper state
        if 'ee_grip_right' in obs:
            state['ee_grip_right'] = obs['ee_grip_right']
        if 'ee_grip_left' in obs:
            state['ee_grip_left'] = obs['ee_grip_left']
        
        return state

    def calculate_state_difference(self, state_before, state_after):
        """Calculate the difference between two robot states as a correction vector"""
        # Initialize correction vector [right_pos(3), right_orn(3), right_grip(1), left_pos(3), left_orn(3), left_grip(1)]
        correction = np.zeros(14)
        
        # If we have qpos, we can calculate the joint-space difference
        # This would be useful for learning a residual policy in joint space
        if 'qpos' in state_before and 'qpos' in state_after:
            joint_diff = state_after['qpos'] - state_before['qpos']
            # Store this for additional analysis, but we'll still use task space for the correction vector
        
        # Right arm position difference (indices 0-2)
        if 'ee_pos_right' in state_before and 'ee_pos_right' in state_after:
            correction[0:3] = state_after['ee_pos_right'] - state_before['ee_pos_right']
        
        # Right arm orientation difference (indices 3-5)
        if 'ee_orn_right' in state_before and 'ee_orn_right' in state_after:
            correction[3:6] = state_after['ee_orn_right'] - state_before['ee_orn_right']
        
        # Right gripper difference (index 6)
        if 'ee_grip_right' in state_before and 'ee_grip_right' in state_after:
            correction[6] = state_after['ee_grip_right'] - state_before['ee_grip_right']
        
        # Left arm position difference (indices 7-9)
        if 'ee_pos_left' in state_before and 'ee_pos_left' in state_after:
            correction[7:10] = state_after['ee_pos_left'] - state_before['ee_pos_left']
        
        # Left arm orientation difference (indices 10-12)
        if 'ee_orn_left' in state_before and 'ee_orn_left' in state_after:
            correction[10:13] = state_after['ee_orn_left'] - state_before['ee_orn_left']
        
        # Left gripper difference (index 13)
        if 'ee_grip_left' in state_before and 'ee_grip_left' in state_after:
            correction[13] = state_after['ee_grip_left'] - state_before['ee_grip_left']
        
        return correction

    def visualize_adjustment(self, policy_action, final_action, save_path):
        """Create a visualization of the adjustment between policy and corrected actions"""
        # Calculate the adjustment
        adjustment = final_action - policy_action
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot position adjustments
        positions = ['X1', 'Y1', 'Z1', 'Roll1', 'Pitch1', 'Yaw1', 'Grip1', 
                    'X2', 'Y2', 'Z2', 'Roll2', 'Pitch2', 'Yaw2', 'Grip2']
        
        # Plot position and orientation adjustments (excluding grippers)
        pos_indices = [0, 1, 2, 7, 8, 9]  # X, Y, Z for both arms
        pos_labels = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
        
        ax1.bar(pos_labels, adjustment[pos_indices])
        ax1.set_ylabel('Adjustment')
        ax1.set_title('Position Adjustments')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot orientation and gripper adjustments
        ori_indices = [3, 4, 5, 6, 10, 11, 12, 13]  # Roll, Pitch, Yaw, Grip for both arms
        ori_labels = ['Roll1', 'Pitch1', 'Yaw1', 'Grip1', 'Roll2', 'Pitch2', 'Yaw2', 'Grip2']
        
        ax2.bar(ori_labels, adjustment[ori_indices])
        ax2.set_ylabel('Adjustment')
        ax2.set_title('Orientation and Gripper Adjustments')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Function to set gripper to a specific state
    def set_gripper(self, gripper_index, state_value, control_data, status_label=None, apply_action_func=None):
        """Optimized gripper control function"""
        # Get current state
        current_state = control_data['base_action'][gripper_index]
        
        # More efficient action creation
        new_action = np.zeros(14)
        new_action[gripper_index] = state_value
        
        # Preserve other gripper state
        other_gripper = 13 if gripper_index == 6 else 6
        new_action[other_gripper] = control_data['base_action'][other_gripper]
        
        # Record correction with minimal data
        control_data['accumulated_corrections'].append({
            'axis': gripper_index,
            'type': 'gripper_set',
            'original_value': current_state,
            'new_value': state_value,
            'timestamp': time.time()
        })
        
        # Update action
        control_data['action'] = new_action
        
        # Update UI efficiently
        if status_label:
            gripper_name = "Left" if gripper_index == 13 else "Right"
            state_text = "OPEN" if state_value > 0 else "CLOSED"
            status_label.config(text=f"{gripper_name} Gripper: {state_text}")
        
        # Apply action with function check
        if callable(apply_action_func):
            apply_action_func()
        else:
            control_data['done'] = True

    def move_x_positive(self, control_data, status_label, apply_action):
        """Move in positive X direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][0] += control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved X+: {control_data['action'][0]:.3f}")
        else:
            # In orientation mode, adjust yaw
            control_data['action'][3] += control_data['step_size']  # yaw for right arm
            status_label.config(text=f"Increased Right Yaw: {control_data['action'][3]:.3f}")
        
        apply_action()

    def move_x_negative(self, control_data, status_label, apply_action):
        """Move in negative X direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][0] -= control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved X-: {control_data['action'][0]:.3f}")
        else:
            # In orientation mode, adjust yaw
            control_data['action'][3] -= control_data['step_size']  # yaw for right arm
            status_label.config(text=f"Decreased Right Yaw: {control_data['action'][3]:.3f}")
        
        apply_action()

    def move_y_positive(self, control_data, status_label, apply_action):
        """Move in positive Y direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][1] += control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Y+: {control_data['action'][1]:.3f}")
        else:
            # In orientation mode, adjust pitch
            control_data['action'][4] += control_data['step_size']  # pitch for right arm
            status_label.config(text=f"Increased Right Pitch: {control_data['action'][4]:.3f}")
        
        apply_action()

    def move_y_negative(self, control_data, status_label, apply_action):
        """Move in negative Y direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][1] -= control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Y-: {control_data['action'][1]:.3f}")
        else:
            # In orientation mode, adjust pitch
            control_data['action'][4] -= control_data['step_size']  # pitch for right arm
            status_label.config(text=f"Decreased Right Pitch: {control_data['action'][4]:.3f}")
        
        apply_action()

    def move_z_positive(self, control_data, status_label, apply_action):
        """Move in positive Z direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][2] += control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Z+: {control_data['action'][2]:.3f}")
        else:
            # In orientation mode, adjust yaw
            control_data['action'][3] += control_data['step_size']  # yaw for right arm
            status_label.config(text=f"Increased Right Yaw: {control_data['action'][3]:.3f}")
        
        apply_action()

    def move_z_negative(self, control_data, status_label, apply_action):
        """Move in negative Z direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][2] -= control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Z-: {control_data['action'][2]:.3f}")
        else:
            # In orientation mode, adjust yaw
            control_data['action'][3] -= control_data['step_size']  # yaw for right arm
            status_label.config(text=f"Decreased Right Yaw: {control_data['action'][3]:.3f}")
        
        apply_action()

    def move_x2_positive(self, control_data, status_label, apply_action):
        """Move left gripper in positive X direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][7] += control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Left X+: {control_data['action'][7]:.3f}")
        else:
            # In orientation mode, adjust roll
            control_data['action'][10] += control_data['step_size']  # roll for left arm
            status_label.config(text=f"Increased Left Roll: {control_data['action'][10]:.3f}")
        
        apply_action()

    def move_x2_negative(self, control_data, status_label, apply_action):
        """Move left gripper in negative X direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][7] -= control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Left X-: {control_data['action'][7]:.3f}")
        else:
            # In orientation mode, adjust roll
            control_data['action'][10] -= control_data['step_size']  # roll for left arm
            status_label.config(text=f"Decreased Left Roll: {control_data['action'][10]:.3f}")
        
        apply_action()

    def move_y2_positive(self, control_data, status_label, apply_action):
        """Move left gripper in positive Y direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][8] += control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Left Y+: {control_data['action'][8]:.3f}")
        else:
            # In orientation mode, adjust pitch
            control_data['action'][11] += control_data['step_size']  # pitch for left arm
            status_label.config(text=f"Increased Left Pitch: {control_data['action'][11]:.3f}")
        
        apply_action()

    def move_y2_negative(self, control_data, status_label, apply_action):
        """Move left gripper in negative Y direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][8] -= control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Left Y-: {control_data['action'][8]:.3f}")
        else:
            # In orientation mode, adjust pitch
            control_data['action'][11] -= control_data['step_size']  # pitch for left arm
            status_label.config(text=f"Decreased Left Pitch: {control_data['action'][11]:.3f}")
        
        apply_action()

    def move_z2_positive(self, control_data, status_label, apply_action):
        """Move left gripper in positive Z direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][9] += control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Left Z+: {control_data['action'][9]:.3f}")
        else:
            # In orientation mode, adjust yaw
            control_data['action'][12] += control_data['step_size']  # yaw for left arm
            status_label.config(text=f"Increased Left Yaw: {control_data['action'][12]:.3f}")
        
        apply_action()

    def move_z2_negative(self, control_data, status_label, apply_action):
        """Move left gripper in negative Z direction"""
        if control_data['control_mode'] == 'position':
            # Store original yaw and gripper state
            original_yaw_right = control_data['action'][3]
            original_yaw_left = control_data['action'][8]
            original_grip_right = control_data['action'][4]
            original_grip_left = control_data['action'][9]
            
            # Apply position change
            control_data['action'][9] -= control_data['step_size']
            
            # Restore yaw and gripper state
            control_data['action'][3] = original_yaw_right  # yaw right
            control_data['action'][4] = original_grip_right  # gripper right
            control_data['action'][8] = original_yaw_left  # yaw left
            control_data['action'][9] = original_grip_left  # gripper left
            
            status_label.config(text=f"Moved Left Z-: {control_data['action'][9]:.3f}")
        else:
            # In orientation mode, adjust yaw
            control_data['action'][12] -= control_data['step_size']  # yaw for left arm
            status_label.config(text=f"Decreased Left Yaw: {control_data['action'][12]:.3f}")
        
        apply_action()

def main():
    # Create evaluator
    evaluator = PolicyEvaluator()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Policy Evaluation')
    parser.add_argument('--wait', action='store_false', help='Wait for user input before each action')
    parser.add_argument('--keyboard', action='store_false', help='Enable keyboard control of the robot')
    parser.add_argument('--step-size', type=float, default=0.05, help='Step size for keyboard control movements')
    args = parser.parse_args()
    
    # Set parameters based on command line arguments
    evaluator.wait_for_user_input = args.wait
    evaluator.keyboard_control = args.keyboard
    evaluator.control_step_size = args.step_size
    
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
