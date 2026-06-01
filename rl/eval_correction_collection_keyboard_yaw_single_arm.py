from surrol.tasks.peg_transfer_rpy_new_foveal import PegTransfer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import gym
import time
import numpy as np
import imageio
from surrol.const import ROOT_DIR_PATH
import surrol.gym
import pickle
import cv2
import torch
from agents.act import ACT_Policy
from typing import Optional, List, Dict
import threading
import tkinter as tk
from functools import partial
# os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
import pybullet as p
import matplotlib.pyplot as plt
import h5py

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
        self.env_name = 'PegTransfer-v0' # Changed from BiPegTransfer-v2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.render = False
        self.mode = 'whole'
        self.save_video = True
        self.video_dir = os.path.expanduser('~/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/eval/whole_policy/0621-stereo-100_correction_single_arm/')
        print("\n=== Initialization ===")
        print(f"Setting video directory to: {self.video_dir}")
        
        # Create directories if they don't exist
        if self.save_video and not os.path.exists(self.video_dir):
            print(f"Creating video directory structure...")
            os.makedirs(self.video_dir, exist_ok=True)
            os.makedirs(os.path.join(self.video_dir, 'success'), exist_ok=True)
            os.makedirs(os.path.join(self.video_dir, 'failed'), exist_ok=True)
            os.makedirs(os.path.join(self.video_dir, 'correction_data'), exist_ok=True)
            print("Directory structure created successfully")
        
        self.wait_for_user_input = True
        self.keyboard_control = True  # New flag for keyboard control
        self.control_step_size = 1  # Step size for keyboard control movements
        
        # Define model paths
        # NOTE: These model paths are for the bimanual task. You might need to update them for the single-arm task.
        self.model_paths = {
            'whole': {
                'dir': '/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/ACT_Peg',
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
        self.env = PegTransfer(render_mode='human',action_mode='rpy')

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

    def gui_control_robot(self, action, accumulated_corrections=None, episode_num=None):
        """Control the robot using a graphical user interface with correction accumulation"""
        # Initialize accumulated_corrections if None
        if accumulated_corrections is None:
            accumulated_corrections = []
        
        # Create the main window
        root = tk.Tk()
        root.title("Robot Control Interface")
        
        # Initialize control data with last_update_time
        control_data = {
            'action': action.copy(),
            'base_action': action.copy(),
            'done': False,
            'exit': False,
            'quit': False,
            'reset': False,  # New flag for reset
            'step_size': self.control_step_size,
            'accumulated_corrections': accumulated_corrections,
            'last_update_time': time.time(),
            'episode_num': episode_num,
            'control_mode': 'position'  # Start in position control mode
        }
        
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
        def exit_control():
            control_data['done'] = True
            control_data['exit'] = True
            root.destroy()
        
        # Function to quit correction mode entirely
        def quit_correction_mode():
            control_data['done'] = True
            control_data['exit'] = True
            control_data['quit'] = True
            root.destroy()
        
        # Optimize action application with throttling
        def apply_action():
            current_time = time.time()
            # Throttle updates to prevent excessive environment steps
            if current_time - control_data['last_update_time'] < 0.1:  # 100ms minimum between actions
                return
            
            control_data['last_update_time'] = current_time
            control_data['done'] = True
            print(f"Applying action: {control_data['action']}")
            root.quit()
        
        # Optimize movement function with better constraint handling
        def move_robot(axis, direction):
            # Create action vector more efficiently
            new_action = np.zeros(7) # Changed from 14 to 7 for single arm
            new_action[axis] = direction * control_data['step_size']
            
            # Explicitly preserve gripper states from the original policy action
            new_action[6] = control_data['base_action'][6]  # Gripper
            
            # Record this correction in the accumulated corrections list
            correction = {
                'axis': axis,
                'direction': direction,
                'step_size': control_data['step_size'],
                'original_value': control_data['action'][axis],
                'new_value': new_action[axis],
                'timestamp': time.time()
            }
            control_data['accumulated_corrections'].append(correction)
            
            # Update the current action
            control_data['action'] = new_action.copy()
            
            # Update status
            status_label.config(text=f"Applied: Axis {axis} {'+' if direction > 0 else '-'}{control_data['step_size']}")
            
            # Apply the action immediately
            apply_action()
        
        # Create main frame
        main_frame = tk.Frame(root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frame for arm control
        arm_frame = tk.LabelFrame(main_frame, text="Arm Control", padx=15, pady=15, font=("Arial", 12, "bold"))
        arm_frame.grid(row=0, column=0, columnspan=2, padx=15, pady=15, sticky="nsew")
        
        # Configure button style
        button_font = ("Arial", 12)
        button_width = 5
        button_height = 2
        label_font = ("Arial", 11)
        
        # Arm frame
        tk.Label(arm_frame, text="Position Control", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, pady=10)

        # X axis controls for arm
        tk.Label(arm_frame, text="X Axis:", font=label_font).grid(row=1, column=0, sticky="e", padx=5)
        tk.Button(arm_frame, text="-", command=lambda: move_robot(0, -1), width=button_width, height=button_height, font=button_font).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(arm_frame, text="+", command=lambda: move_robot(0, 1), width=button_width, height=button_height, font=button_font).grid(row=1, column=2, padx=5, pady=5)

        # Y axis controls for arm
        tk.Label(arm_frame, text="Y Axis:", font=label_font).grid(row=2, column=0, sticky="e", padx=5)
        tk.Button(arm_frame, text="-", command=lambda: move_robot(1, -1), width=button_width, height=button_height, font=button_font).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(arm_frame, text="+", command=lambda: move_robot(1, 1), width=button_width, height=button_height, font=button_font).grid(row=2, column=2, padx=5, pady=5)

        # Z axis controls for arm
        tk.Label(arm_frame, text="Z Axis:", font=label_font).grid(row=3, column=0, sticky="e", padx=5)
        tk.Button(arm_frame, text="-", command=lambda: move_robot(2, -1), width=button_width, height=button_height, font=button_font).grid(row=3, column=1, padx=5, pady=5)
        tk.Button(arm_frame, text="+", command=lambda: move_robot(2, 1), width=button_width, height=button_height, font=button_font).grid(row=3, column=2, padx=5, pady=5)

        # Create rotation control section for arm
        tk.Label(arm_frame, text="Rotation Control", font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=3, pady=10)

        # Roll controls for arm
        tk.Label(arm_frame, text="Roll:", font=label_font).grid(row=5, column=0, sticky="e", padx=5)
        tk.Button(arm_frame, text="-", command=lambda: move_robot(3, -1), width=button_width, height=button_height, font=button_font).grid(row=5, column=1, padx=5, pady=5)
        tk.Button(arm_frame, text="+", command=lambda: move_robot(3, 1), width=button_width, height=button_height, font=button_font).grid(row=5, column=2, padx=5, pady=5)

        # Pitch controls for arm
        tk.Label(arm_frame, text="Pitch:", font=label_font).grid(row=6, column=0, sticky="e", padx=5)
        tk.Button(arm_frame, text="-", command=lambda: move_robot(4, -1), width=button_width, height=button_height, font=button_font).grid(row=6, column=1, padx=5, pady=5)
        tk.Button(arm_frame, text="+", command=lambda: move_robot(4, 1), width=button_width, height=button_height, font=button_font).grid(row=6, column=2, padx=5, pady=5)

        # Yaw controls for arm
        tk.Label(arm_frame, text="Yaw:", font=label_font).grid(row=7, column=0, sticky="e", padx=5)
        tk.Button(arm_frame, text="-", command=lambda: move_robot(5, -1), width=button_width, height=button_height, font=button_font).grid(row=7, column=1, padx=5, pady=5)
        tk.Button(arm_frame, text="+", command=lambda: move_robot(5, 1), width=button_width, height=button_height, font=button_font).grid(row=7, column=2, padx=5, pady=5)

        # For arm
        # Create a frame for gripper controls
        gripper_frame = tk.Frame(arm_frame)
        gripper_frame.grid(row=8, column=0, columnspan=3, pady=10)

        # Open gripper button
        tk.Button(gripper_frame, text="Open Gripper", 
                 command=lambda: self.set_gripper(6, 0.5, control_data, status_label, apply_action),
                 width=10, height=2, font=button_font, bg="#aaffaa").pack(side=tk.LEFT, padx=5)

        # Close gripper button
        tk.Button(gripper_frame, text="Close Gripper", 
                 command=lambda: self.set_gripper(6, -0.5, control_data, status_label, apply_action),
                 width=10, height=2, font=button_font, bg="#ffaaaa").pack(side=tk.LEFT, padx=5)
        
        # Create control panel
        control_frame = tk.LabelFrame(main_frame, text="Control Panel", padx=15, pady=15, font=("Arial", 12, "bold"))
        control_frame.grid(row=1, column=0, columnspan=2, padx=15, pady=15, sticky="ew")
        
        # Step size controls
        step_frame = tk.Frame(control_frame)
        step_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(step_frame, text="Step Size:", font=label_font).pack(side=tk.LEFT, padx=10)
        tk.Button(step_frame, text="-", command=lambda: adjust_step_size(-0.1), width=3, height=1, font=button_font).pack(side=tk.LEFT, padx=10)
        step_size_label = tk.Label(step_frame, text=f"Step Size: {control_data['step_size']:.2f}", font=label_font)
        step_size_label.pack(side=tk.LEFT, padx=10)
        tk.Button(step_frame, text="+", command=lambda: adjust_step_size(0.1), width=3, height=1, font=button_font).pack(side=tk.LEFT, padx=10)
        
        # Function to adjust step size
        def adjust_step_size(delta):
            """Adjust the step size for movements"""
            # Update step size with bounds checking (minimum 0.01, maximum 1.0)
            control_data['step_size'] = max(0.01, min(1.0, control_data['step_size'] + delta))
            # Update the label
            step_size_label.config(text=f"Step Size: {control_data['step_size']:.2f}")
            # Update status
            status_label.config(text=f"Step size adjusted to: {control_data['step_size']:.2f}")
        
        # Add reset button to control panel
        def add_reset_button(control_frame):
            reset_button = tk.Button(
                control_frame,
                text="Reset Episode",
                command=reset_episode,
                width=15,
                height=2,
                font=("Arial", 12),
                bg="#ff9999",  # Light red background
                fg="white"
            )
            reset_button.pack(side=tk.LEFT, padx=10)
        
        # Action buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=15)
        
        tk.Button(button_frame, text="Reset Action", command=exit_control, width=15, height=2, font=button_font).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Quit Correction", command=quit_correction_mode, width=15, height=2, font=button_font).pack(side=tk.LEFT, padx=10)
        add_reset_button(button_frame)  # Add the reset button
        
        # Add a new Quit button with a distinctive color
        quit_button = tk.Button(button_frame, text="Quit Correction Mode", command=quit_correction_mode, 
                               width=20, height=2, font=button_font, bg="#ff6666", fg="white")
        quit_button.pack(side=tk.RIGHT, padx=10)
        
        # Add a mode toggle for position vs. orientation control
        control_data['control_mode'] = 'position'  # Start in position control mode
        
        # Add mode toggle button
        def toggle_control_mode():
            if control_data['control_mode'] == 'position':
                control_data['control_mode'] = 'orientation'
                mode_button.config(text="Mode: ORIENTATION", bg="#ffaa66")
                status_label.config(text="Switched to ORIENTATION control mode. Arrow keys now control orientation.")
            else:
                control_data['control_mode'] = 'position'
                mode_button.config(text="Mode: POSITION", bg="#66aaff")
                status_label.config(text="Switched to POSITION control mode. Arrow keys now control position.")
        
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
        
        # Status display
        status_frame = tk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=2, padx=15, pady=10, sticky="ew")
        
        status_label = tk.Label(status_frame, text="Ready for corrections. Each button press applies a single correction.", 
                               wraplength=800, justify=tk.LEFT, font=("Arial", 11))
        status_label.pack(fill=tk.X)
        
        # Add view control buttons
        view_frame = tk.Frame(main_frame)
        view_frame.grid(row=3, column=0, columnspan=2, padx=15, pady=5, sticky="ew")
        
        # Define view monitoring functions first
        def change_camera_view(view_type):
            """Change the camera view in the PyBullet simulator"""
            try:
                # Get the scaling from the environment
                scaling = self.env.SCALING if hasattr(self.env, 'SCALING') else 5.0
                
                # Make sure PyBullet is connected
                if p.getConnectionInfo()['isConnected']:
                    if view_type == "top":
                        # Set top-down view focused on the table
                        p.resetDebugVisualizerCamera(
                            cameraDistance=0.4 * scaling,  # Closer to see details
                            cameraYaw=90,
                            cameraPitch=-75,  # Less extreme angle to see depth
                            cameraTargetPosition=(2.2, 0, 2.5)  # Target the peg board surface
                        )
                        status_label.config(text="Changed to top-down view")
                    elif view_type == "front":
                        # Set front view
                        p.resetDebugVisualizerCamera(
                            cameraDistance=0.9 * scaling,
                            cameraYaw=90,
                            cameraPitch=-30,
                            cameraTargetPosition=(0.55, 0, 0.7)
                        )
                        status_label.config(text="Changed to front view")
                    elif view_type == "side":
                        # Set side view
                        p.resetDebugVisualizerCamera(
                            cameraDistance=0.9 * scaling,
                            cameraYaw=0,
                            cameraPitch=-30,
                            cameraTargetPosition=(0.55, 0, 0.7)
                        )
                        status_label.config(text="Changed to side view")
                    
                    # Force GUI update
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                else:
                    status_label.config(text="PyBullet connection not available")
            except Exception as e:
                status_label.config(text=f"Camera view error: {str(e)}")
                print(f"Error changing camera view: {str(e)}")
        
        def start_view_monitor():
            """Start monitoring camera parameters"""
            if not hasattr(control_data, 'view_monitor_active'):
                control_data['view_monitor_active'] = True
                update_view_monitor()
                status_label.config(text="View monitoring started. Move camera with mouse to see parameters.")
            else:
                control_data['view_monitor_active'] = not control_data['view_monitor_active']
                status_label.config(text=f"View monitoring {'started' if control_data['view_monitor_active'] else 'stopped'}")

        def update_view_monitor():
            """Update the display of current camera parameters"""
            if hasattr(control_data, 'view_monitor_active') and control_data['view_monitor_active']:
                try:
                    # Get current camera parameters
                    cam_data = p.getDebugVisualizerCamera()
                    
                    # Print the raw data to understand its structure
                    print("Raw camera data:", cam_data)
                    
                    # Extract camera parameters safely
                    if len(cam_data) >= 11:  # Make sure we have enough elements
                        width, height = cam_data[0], cam_data[1]
                        view_matrix = cam_data[2]
                        projection_matrix = cam_data[3]
                        
                        # These indices might need adjustment based on the actual structure
                        if len(cam_data) >= 14:
                            distance = cam_data[10]
                            yaw = cam_data[8]
                            pitch = cam_data[9]
                            target_pos = (cam_data[11], cam_data[12], cam_data[13])
                        else:
                            # If we don't have all expected fields, use what we can get
                            distance = cam_data[-4] if len(cam_data) > 4 else 0
                            yaw = cam_data[-6] if len(cam_data) > 6 else 0
                            pitch = cam_data[-5] if len(cam_data) > 5 else 0
                            target_pos = (0, 0, 0)  # Default if not available
                            
                        # Display in status
                        status_label.config(text=f"Saved view: Distance={distance:.2f}, Yaw={yaw:.1f}, "
                                           f"Pitch={pitch:.1f}, Target={target_pos}")
                        
                        # Add to history
                        view_info = (f"Camera view saved: Distance={distance:.2f}, Yaw={yaw:.1f}, "
                                    f"Pitch={pitch:.1f}, Target={target_pos}")
                        history_text.insert(tk.END, view_info + "\n")
                        
                        # Print to console for easy copying
                        print("\n=== CAMERA VIEW PARAMETERS ===")
                        print(f"Distance: {distance}")
                        print(f"Yaw: {yaw}")
                        print(f"Pitch: {pitch}")
                        print(f"Target Position: {target_pos}")
                        print("==============================\n")
                        
                        # Optionally save to a file
                        with open(os.path.join(self.video_dir, 'camera_views.txt'), 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {view_info}\n")
                    else:
                        status_label.config(text="Camera data format unexpected")
                        print("Camera data format unexpected:", cam_data)
                    
                except Exception as e:
                    status_label.config(text=f"Error monitoring view: {str(e)}")
                    control_data['view_monitor_active'] = False

        def save_current_view():
            """Save the current camera parameters to a file and display them"""
            try:
                # Get current camera parameters
                cam_data = p.getDebugVisualizerCamera()
                
                # Print the raw data to understand its structure
                print("Raw camera data:", cam_data)
                
                # Extract camera parameters safely
                if len(cam_data) >= 11:  # Make sure we have enough elements
                    width, height = cam_data[0], cam_data[1]
                    view_matrix = cam_data[2]
                    projection_matrix = cam_data[3]
                    
                    # These indices might need adjustment based on the actual structure
                    if len(cam_data) >= 14:
                        distance = cam_data[10]
                        yaw = cam_data[8]
                        pitch = cam_data[9]
                        target_pos = (cam_data[11], cam_data[12], cam_data[13])
                    else:
                        # If we don't have all expected fields, use what we can get
                        distance = cam_data[-4] if len(cam_data) > 4 else 0
                        yaw = cam_data[-6] if len(cam_data) > 6 else 0
                        pitch = cam_data[-5] if len(cam_data) > 5 else 0
                        target_pos = (0, 0, 0)  # Default if not available
                        
                    # Display in status
                    status_label.config(text=f"Saved view: Distance={distance:.2f}, Yaw={yaw:.1f}, "
                                       f"Pitch={pitch:.1f}, Target={target_pos}")
                    
                    # Add to history
                    view_info = (f"Camera view saved: Distance={distance:.2f}, Yaw={yaw:.1f}, "
                                f"Pitch={pitch:.1f}, Target={target_pos}")
                    history_text.insert(tk.END, view_info + "\n")
                    
                    # Print to console for easy copying
                    print("\n=== CAMERA VIEW PARAMETERS ===")
                    print(f"Distance: {distance}")
                    print(f"Yaw: {yaw}")
                    print(f"Pitch: {pitch}")
                    print(f"Target Position: {target_pos}")
                    print("==============================\n")
                    
                    # Optionally save to a file
                    with open(os.path.join(self.video_dir, 'camera_views.txt'), 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {view_info}\n")
                else:
                    status_label.config(text="Camera data format unexpected")
                    print("Camera data format unexpected:", cam_data)
                
            except Exception as e:
                status_label.config(text=f"Error saving view: {str(e)}")
                print(f"Error saving view: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Now create the buttons after the functions are defined
        tk.Label(view_frame, text="Camera Views:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(view_frame, text="Top-Down View", command=lambda: change_camera_view("top"), 
                 width=15, height=1, font=button_font).pack(side=tk.LEFT, padx=10)
        tk.Button(view_frame, text="Front View", command=lambda: change_camera_view("front"), 
                 width=15, height=1, font=button_font).pack(side=tk.LEFT, padx=10)
        tk.Button(view_frame, text="Side View", command=lambda: change_camera_view("side"), 
                 width=15, height=1, font=button_font).pack(side=tk.LEFT, padx=10)
        tk.Button(view_frame, text="Monitor View", command=start_view_monitor, 
                 width=15, height=1, font=button_font, bg="#aaddff").pack(side=tk.LEFT, padx=10)
        tk.Button(view_frame, text="Save View", command=save_current_view, 
                 width=15, height=1, font=button_font, bg="#aaffaa").pack(side=tk.LEFT, padx=10)
        
        # Correction history display
        history_frame = tk.LabelFrame(main_frame, text="Correction History", padx=15, pady=15, font=("Arial", 12, "bold"))
        history_frame.grid(row=4, column=0, columnspan=2, padx=15, pady=15, sticky="ew")
        
        history_text = tk.Text(history_frame, height=5, width=80, font=("Arial", 10))
        history_text.pack(fill=tk.BOTH, expand=True)
        
        # Function to update correction history
        def update_history():
            """Update the correction history display"""
            history_text.delete(1.0, tk.END)
            for i, corr in enumerate(control_data['accumulated_corrections']):
                if 'type' in corr and corr['type'] in ['gripper_toggle', 'gripper_set']:
                    # For gripper operations, show the change with the new open/closed logic
                    state_before = "OPEN" if corr['original_value'] > 0 else "CLOSED"
                    state_after = "OPEN" if corr['new_value'] > 0 else "CLOSED"
                    gripper_name = "Arm"
                    history_text.insert(tk.END, f"{i+1}. {gripper_name} Gripper set from {state_before} to {state_after}\n")
                elif 'direction' in corr:
                    # For movement corrections
                    history_text.insert(tk.END, f"{i+1}. Axis {corr['axis']} adjusted by " +
                                       f"{'+' if corr['direction'] > 0 else '-'}{corr['step_size']}\n")
                else:
                    # For any other type of correction
                    history_text.insert(tk.END, f"{i+1}. Other correction applied\n")
        
        # Set window size and position
        root.geometry("900x900")  # Larger window size to accommodate history
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # Run the GUI in a loop to handle immediate action application
        while not control_data['exit']:
            control_data['done'] = False
            root.mainloop()  # This will block until quit() is called
            
            # If we have an action to apply, return it to the caller
            if control_data['done'] and not control_data['exit']:
                # Update the correction history
                update_history()
                
                # Return the action to be applied
                action_to_apply = control_data['action'].copy()
                
                # Let the caller apply the action and update the environment
                yield action_to_apply
                
                # Continue the GUI loop
                if not control_data['exit']:
                    root.deiconify()  # Make sure the window is visible
                    root.mainloop()
        
        # Save the correction data
        # if len(control_data['accumulated_corrections']) > 0:
        #     self.save_corrections(control_data['accumulated_corrections'], control_data['episode_num'])
        
        # Return a special value to signal quitting correction mode
        if control_data['quit']:
            return "QUIT"
        
        # At the end of the generator loop, check for reset
        if control_data['reset']:
            return "RESET"
        
        # Return None to signal normal exit from control mode
        return None

    def interactive_control_loop(self, obs, episode_length=0):
        """Optimized interactive control loop with better performance"""
        success = False
        image_list = []
        text_list = []
        
        # Pre-allocate memory for gripper states
        gripper_states = np.zeros(1) # For single arm
        
        while not success and episode_length < 200:
            # Get policy action more efficiently
            with torch.no_grad():
                action = self.act_class1.get_action(obs, episode_length)
            
            # Update gripper states efficiently
            gripper_states[0] = action[6]   # Gripper
            
            # Apply policy action
            obs, reward, done, info = self.env.step(action)
            
            # Efficient image handling
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
            
            # Check for user modification
            modify = input("\nModify pose? (m/enter): ").lower() == 'm'
            
            if modify:
                # Create control action efficiently
                raw_action = np.zeros(7) # For single arm
                raw_action[6] = gripper_states[0]
                
                # Process GUI control actions efficiently
                try:
                    for control_action in self.gui_control_robot(raw_action, [], episode_length):
                        if control_action == "QUIT":
                            break
                        
                        # Apply action with proper verification
                        action_to_apply = self.verify_action_format(control_action.copy())
                        obs, reward, done, info = self.env.step(action_to_apply)
                        
                        # Check for success
                        if info['is_success'] > 0:
                            success = True
                            break
                except StopIteration:
                    pass
            
            # Check for success and update counters
            if info['is_success'] > 0:
                success = True
            
            episode_length += 1
            text_list.append('interactive control')
        
        return obs, success, image_list, text_list, episode_length

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
        """Collect data for robot corrections with specific data structure and videos"""
        print("\n=== Data Collection Setup ===")
        print(f"Base video directory: {self.video_dir}")
        
        # Create directories if they don't exist
        episode_dir = os.path.join(self.video_dir, 'correction_data')
        os.makedirs(episode_dir, exist_ok=True)
        
        while True:  # Loop to allow episode resets
            # Initialize episode
            print("\n=== Starting new episode - resetting environment ===")
            obs = self.env.reset()
            print("=== Environment reset complete ===\n")
            
            # Initialize data collection arrays with correct shapes
            data = {
                "is_human_intervention": [],      # shape (L,)
                "policy_action": [],              # shape (L, 7)
                "policy_obs": [],                 # shape (L, 26)
                "post_intervention_eef_pose": [], # shape (L, 7)
                "pre_intervention_eef_pose": [],  # shape (L, 7)
                'correction_action': [],
                'images': []
            }
            
            # Initialize video collection
            video_frames = []
            text_overlay = []
            episode_length = 0
            success = False
            reset_requested = False
            
            while not success and episode_length < 200:
                # Store video frame
                if 'images' in obs:
                    video_frames.append(obs['images'])
                else:
                    video_frames.append({'rgb1': obs['image']})
                
                # Get policy action prediction
                with torch.no_grad():
                    policy_action = self.act_class1.get_action(obs, episode_length)

                observation = obs['observation']
                robot_state = obs['qpos']
                images = obs['images']

                data['images'].append(images)
                
                # Store policy action and observation
                data["policy_action"].append(policy_action)
                data["policy_obs"].append(observation)
                ##############################
                ###### apply act policy ######
                ##############################
                # Add text overlay for policy
                text_overlay.append("Policy Action")
                
                # For non-intervention steps, store the current state
                data["pre_intervention_eef_pose"].append(robot_state)
                
                # Apply policy action
                obs, reward, done, info = self.env.step(policy_action)
                
                # Ask for correction
                print("\nCurrent robot pose applied. Do you want to modify it?")
                modify = input("Press 'm' to modify, any other key to continue: ").lower()
                
                needs_correction = modify == 'm'
                data["is_human_intervention"].append(1 if needs_correction else 0)
                total_correction = np.zeros(7)
                if needs_correction:
                    # Add text overlay for intervention
                    text_overlay.append("Human Intervention")
                    # 
                    # Get corrections through GUI
                    final_obs = None
                    for control_action in self.gui_control_robot(policy_action, accumulated_corrections=[], episode_num=episode_num):
                        if control_action == "RESET":
                            reset_requested = True
                            break
                        if control_action == "QUIT":
                            return False, episode_length
                        
                        final_obs, reward, done, info = self.env.step(control_action)
                        total_correction += control_action
                    
                    if reset_requested:
                        break
                    
                    obs = final_obs  # Update obs with final state after intervention
                    observation = obs['observation']
                    robot_state = obs['qpos']
                    
                    # Store post-intervention state
                    data["post_intervention_eef_pose"].append(robot_state)
                    data['correction_action'].append(total_correction)
                else:
                    data["post_intervention_eef_pose"].append(np.zeros_like(robot_state))
                    data['correction_action'].append(np.zeros_like(total_correction))
                    
                
                # Check for success
                if info['is_success'] > 0:
                    success = True
                    print("Success!")
                
                episode_length += 1
            
            # If reset was requested, start a new episode
            if reset_requested:
                continue
            
            # Only save data if episode was successful
            if success:
                # Create episode-specific directory
                episode_path = os.path.join(episode_dir, f'episode_{episode_num:03d}')
                os.makedirs(episode_path, exist_ok=True)
                
                # Save HDF5 data
                h5_path = os.path.join(episode_path, 'data.h5')
                print(f"\n=== Saving Data ===")
                print(f"Attempting to save to: {h5_path}")
                
                try:
                    with h5py.File(h5_path, 'w') as f:
                        # Convert lists to numpy arrays and save
                        for key in data:
                            if key == "images":
                                # Handle images separately
                                images_grp = f.create_group("images")
                                for i, img_dict in enumerate(data["images"]):
                                    img_step = images_grp.create_group(f'step_{i}')
                                    for cam_name, img in img_dict.items():
                                        img_step.create_dataset(cam_name, data=img)
                            elif key == "policy_obs":
                                # Handle policy observations (numpy arrays)
                                obs_array = np.array(data["policy_obs"])
                                f.create_dataset("policy_obs", data=obs_array)
                            else:
                                # Convert other data to numpy arrays and save
                                array_data = np.array(data[key])
                                print(f"Saving {key} with shape {array_data.shape}")
                                f.create_dataset(key, data=array_data)
                        
                        # Add metadata
                        f.attrs['success'] = success
                        f.attrs['episode_length'] = episode_length
                        f.attrs['timestamp'] = time.time()
                    
                    print(f"Successfully saved data to {h5_path}")
                    
                    # Save video
                    video_path = os.path.join(episode_path, 'episode.mp4')
                    save_videos(video_frames, dt=0.1, video_path=video_path, text_list=text_overlay)
                    print(f"Successfully saved video to {video_path}")
                    
                except Exception as e:
                    print(f"ERROR saving data: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Episode failed - no data saved")
            
            return success, episode_length

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
        # Initialize correction vector for single arm [pos(3), orn(3), grip(1)]
        correction = np.zeros(7)
        
        # If we have qpos, we can calculate the joint-space difference
        # This would be useful for learning a residual policy in joint space
        if 'qpos' in state_before and 'qpos' in state_after:
            joint_diff = state_after['qpos'] - state_before['qpos']
            # Store this for additional analysis, but we'll still use task space for the correction vector
        
        # Arm position difference (indices 0-2)
        if 'ee_pos_right' in state_before and 'ee_pos_right' in state_after:
            correction[0:3] = state_after['ee_pos_right'] - state_before['ee_pos_right']
        
        # Arm orientation difference (indices 3-5)
        if 'ee_orn_right' in state_before and 'ee_orn_right' in state_after:
            correction[3:6] = state_after['ee_orn_right'] - state_before['ee_orn_right']
        
        # Gripper difference (index 6)
        if 'ee_grip_right' in state_before and 'ee_grip_right' in state_after:
            correction[6] = state_after['ee_grip_right'] - state_before['ee_grip_right']
        
        return correction

    def visualize_adjustment(self, policy_action, final_action, save_path):
        """Create a visualization of the adjustment between policy and corrected actions"""
        # Calculate the adjustment
        adjustment = final_action - policy_action
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot position adjustments
        positions = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Grip']
        
        # Plot position and orientation adjustments (excluding grippers)
        pos_indices = [0, 1, 2]  # X, Y, Z
        pos_labels = ['X', 'Y', 'Z']
        
        ax1.bar(pos_labels, adjustment[pos_indices])
        ax1.set_ylabel('Adjustment')
        ax1.set_title('Position Adjustments')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot orientation and gripper adjustments
        ori_indices = [3, 4, 5, 6]  # Roll, Pitch, Yaw, Grip
        ori_labels = ['Roll', 'Pitch', 'Yaw', 'Grip']
        
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
        new_action = np.zeros(7) # For single arm
        new_action[gripper_index] = state_value
        
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
            gripper_name = "Arm"
            state_text = "OPEN" if state_value > 0 else "CLOSED"
            status_label.config(text=f"{gripper_name} Gripper: {state_text}")
        
        # Apply action with function check
        if callable(apply_action_func):
            apply_action_func()
        else:
            control_data['done'] = True

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
