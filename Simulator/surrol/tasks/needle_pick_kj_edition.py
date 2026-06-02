import os
import time
import numpy as np

import pybullet as p
from Simulator.surrol.tasks.psm_env_rpy import PsmEnv
from Simulator.surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,    
    wrap_angle
)
from Simulator.surrol.tasks.ecm_env import EcmEnv, goal_distance
import matplotlib.pyplot as plt
from Simulator.surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from Simulator.surrol.const import ASSET_DIR_PATH
from Simulator.surrol.robots.ecm import Ecm

import pickle
import collections
# NeedlePick 继承了 PsmEnv 类，扩展了环境行为，
# 设计一个专门用于针取任务的模拟环境
class NeedlePick(PsmEnv):
    # 托盘的位姿
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    # 机器人允许操作的空间范围
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.
    # 初始化关节位置
    QPOS_ECM = (0, 0.6, 0.15, 0)
    ### add distance threshold for needle pick task
    DISTANCE_THRESHOLD = 0.01

    ACTION_ECM_SIZE=3
    # 是否启用触觉反馈
    haptic=True
    counter=0
    img_list={}
    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode='human', cid = -1, action_mode = 'yaw'):
        # render_mode:控制环境是否渲染图像
        super(NeedlePick, self).__init__(render_mode, cid, action_mode)
        self.counter = 0
        self.image_list = []
        self.actions_list = []

        # sky
        self.qpos_list = []
        self.prev_qpos = None
        self.episode_idx = 0
        self.new_actions = []
        self.last_action_count = 0


    def _env_setup(self):
        super(NeedlePick, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True
 
        # camera
        if self._render_mode == 'human':

            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        # 初始化一个 ECM（内窥镜相机），指定初始位置和缩放比例。       
        self.ecm = Ecm((0.15, 0.0, 0.8524), 
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)
        self.STEREO=True  # TODO

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # 加载托盘和针
        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle - controlled randomization
        # Moderate rotation randomness: ±45 degrees 
        yaw = (np.random.rand() - 0.5) * np.pi/2  # ±π/4 = ±45 degrees
        
        # Add more position randomization while keeping it reasonable
        # Spawn in a broader area but avoid extremes
        spawn_x = workspace_limits[0].mean() - 0.01 + (np.random.rand() - 0.5) * 0.06  # center-left ± 0.03
        spawn_y = workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.06  # center ± 0.03
        
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm_RL.urdf'),
                            (spawn_x, spawn_y, workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

    # 目标采样
    # 在工作空间内随机采样目标位置
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        # pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        # goal=np.array([pos_obj[0], pos_obj[1],pos_obj[2] + (-0.0007 + 0.0102 + 0.005)])
        
        # Add randomization to goal position while keeping it on the side
        # Base position offset from center to avoid obstructing observation
        base_x = workspace_limits[0][0] + 0.02  # Start at 0.52
        base_y = workspace_limits[1][1] - 0.01  # Start at 0.04
        
        # Add controlled randomization around the base position
        goal_x = base_x + (np.random.rand() - 0.5) * 0.04  # ±0.02 variation
        goal_y = base_y + (np.random.rand() - 0.5) * 0.06  # ±0.03 variation
        goal_z = workspace_limits[2][1] - 0.06 * self.SCALING + (np.random.rand() - 0.5) * 0.02 * self.SCALING  # ±0.01*SCALING variation
        
        # Ensure goal stays within workspace bounds
        goal_x = np.clip(goal_x, workspace_limits[0][0] + 0.01, workspace_limits[0][1] - 0.01)
        goal_y = np.clip(goal_y, workspace_limits[1][0] + 0.01, workspace_limits[1][1] - 0.01)
        
        goal = np.array([goal_x, goal_y, goal_z])
        return goal.copy()
        

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        ### add waypoints for needle pick task ###
        self._waypoints = [None, None, None, None]  # four waypoints
        # self._waypoints = [None]
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)

        roll = orn[0] - np.deg2rad(90)
        pitch = orn[1] if abs(wrap_angle(orn[1] - orn_eef[1])) < abs(wrap_angle(orn[1] + np.pi - orn_eef[1])) \
            else wrap_angle(orn[1] + np.pi)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING,roll, pitch, yaw, 0.5])  # approach (x,y,z,yaw,gripper)
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.010) * self.SCALING, roll,pitch,yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.010) * self.SCALING, roll,pitch,yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING,roll,pitch, yaw, -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    # 观察获取
    # 获取机器人的当前状态、目标位置和相机图像，打包成字典obs返回
    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        # robot_joint_state = self.psm1.get_current_joint_position()
       
        pos, _ = get_link_pose(self.obj_id, -1)
        object_pos = np.array(pos)
        #print("ori obejct pose: ",object_pos)
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)
        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        object_rel_pos = object_pos - robot_state[0: 3]
        
        # tip position
        achieved_goal = np.array(get_link_pose(self.obj_id, self.obj_link1)[0])
            
        ### observation[-6:] = robot_joint_state is the robot joint state at this step ###
        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel(), robot_state  # achieved_goal.copy(),
        ])

        # Render the images (using the ECM)
        output = self.ecm.render_image(stereo=self.STEREO)
        seg = output.mask1
        rgb1 = output.rgb1
        depth = output.depth1
        # obs = {
        #     'observation': observation.copy(),
        #     'achieved_goal': achieved_goal.copy(),
        #     'desired_goal': self.goal.copy()
        # }
        obs = collections.OrderedDict()

        obs['observation'] = observation.copy()
        obs['achieved_goal'] = achieved_goal.copy()
        obs['desired_goal'] = self.goal.copy()

        # sky
        obs['qpos'] = robot_state
        obs['env_state'] = robot_state
        obs['images'] = dict()
        obs['images']['rgb1']= rgb1
        if self.STEREO:
            rgb2 = output.rgb2
            obs['images']['rgb2']= rgb2
        # sky
        obs['images']['mask'] = seg
        obs['images']['depth'] = depth
        #
        obs['observation'] = observation.copy()
        obs['achieved_goal'] = achieved_goal.copy()
        obs['desired_goal'] = self.goal.copy()
        #

        if self.counter==0:
            self.counter+=1
            return obs

        assert rgb1.shape[-1] == 3

        # seg=np.array(seg==6).astype(int)
        
        # if self.counter % 1000 == 0:
        #     self.save_data()
        return obs

    # 动作推断
    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy. I have moved the image collection function here. In the policy testing phase, we will only need the image and write a new function for action inference.
        """
        # robot_joint_state = self.psm1.get_current_joint_position()
        robot_state = self._get_robot_state(idx=0)
        # print(f"joint:,{robot_joint_state}")
        output = self.ecm.render_image(stereo=self.STEREO)
        mask=output.mask1
        rgb1 = output.rgb1
        depth = output.depth1
        assert rgb1.shape[-1] == 3
        ### Edit here for iteratively saving images per frame ###
        # plt.imsave(f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/SKJ-SurRoL-Development/surrol/data/rgb_array1_{self.counter}.png', rgb1)
        if self.STEREO:
            rgb2 = output.rgb2
        self.counter += 1
        # print("CNT:", self.counter)
        action = np.zeros(7)
        action[6] = -0.5  # ?

        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            # delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING

            # Calculate the direction vector from current observation to the waypoint
            direction = waypoint[:3] - obs['observation'][:3]
            # Calculate the distance
            distance = np.linalg.norm(direction)

            # Set delta_pos to the normalized direction scaled by the step size
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_pitch = (waypoint[4] - obs['observation'][4])
            delta_roll = (waypoint[3] - obs['observation'][3])
            
            delta_yaw = (waypoint[5] - obs['observation'][5]).clip(-0.15, 0.15)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2],delta_roll,delta_pitch, delta_yaw, waypoint[6]])
            if i == 2:
                if self.psm1.get_current_jaw_position() != 0 and np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                   self._waypoints[i] = None 
            elif i == 3:
                if self.psm1.get_current_jaw_position() != 0 and np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                    if self.last_action_count > 10:
                        self._waypoints[i] = None
                    else:
                        self.last_action_count += 1
            else:   
                if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                    self._waypoints[i] = None
            break

        return action, rgb1, rgb2, mask, depth, robot_state, i
    
    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        # return np.logical_and(d < self.distance_threshold, self._waypoints[3] is None).astype(np.float32)
        return d < self.distance_threshold

if __name__ == "__main__":
    env = NeedlePick(render_mode='human')  # create one process and corresponding env

    while len(env.image_list) < 10000:
        env.reset()
        env.test()
    env.close()
    # try:
    #     while len(env.image_list) < 10000:
    #         env.reset()
    #         env.test()
    # finally:
    #     print("Closing environment...")
    #     env.close()
    #     p.disconnect()  # 确保断开与 PyBullet 的连接