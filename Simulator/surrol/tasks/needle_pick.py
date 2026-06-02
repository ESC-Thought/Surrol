import os
import time
import numpy as np

import pybullet as p
from Simulator.surrol.tasks.psm_env import PsmEnv
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
    
    ACTION_ECM_SIZE=3
    # 是否启用触觉反馈
    haptic=True
    counter=0
    img_list={}
    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode='human', cid = -1):
        # render_mode:控制环境是否渲染图像
        super(NeedlePick, self).__init__(render_mode, cid)
        self.counter = 0
        self.image_list = []
        self.actions_list = []

        # sky
        self.qpos_list = []
        self.prev_qpos = None
        self.episode_idx = 0
        self.new_actions = []


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
        self.STEREO=True

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

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm_RL.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
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
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        # self._waypoints = [None, None, None, None]  # four waypoints
        self._waypoints = [None]
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach (x,y,z,yaw,gripper)
        # self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
        #                                pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        # self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
        #                                pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        # self._waypoints[3] = np.array([self.goal[0], self.goal[1],
        #                                self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up

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
        robot_joint_state = self.psm1.get_current_joint_position()
       
        pos, _ = get_link_pose(self.obj_id, -1)
        object_pos = np.array(pos)
        #print("ori obejct pose: ",object_pos)
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)
        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        object_rel_pos = object_pos - robot_state[0: 3]
        
        # tip position
        achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
            
        ### observation[-6:] = robot_joint_state is the robot joint state at this step ###
        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel(), robot_joint_state  # achieved_goal.copy(),
        ])

        # Render the images (using the ECM)
        output = self.ecm.render_image(stereo=self.STEREO)
        seg = output.mask1
        rgb1 = output.rgb1
        # obs = {
        #     'observation': observation.copy(),
        #     'achieved_goal': achieved_goal.copy(),
        #     'desired_goal': self.goal.copy()
        # }
        obs = collections.OrderedDict()
        obs['qpos'] = robot_joint_state
        obs['env_state'] = robot_state
        obs['images'] = dict()
        obs['images']['rgb1']= rgb1
        if self.STEREO:
            rgb2 = output.rgb2
            obs['images']['rgb2']= rgb2
        obs['observation'] = observation.copy()
        obs['achieved_goal'] = achieved_goal.copy()
        obs['desired_goal'] = self.goal.copy()

        if self.counter==0:
            self.counter+=1
            return obs
        # sky
        # obs = collections.OrderedDict()
        # obs['qpos'] = self.get_qpos(physics)
        # obs['qvel'] = self.get_qvel(physics)
        # obs['env_state'] = self.get_env_state(physics)
        # obs['images'] = dict()
        # obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        # sky
        # output = self.ecm.render_image(stereo=self.STEREO)
        # seg=output.mask1
        # rgb1=output.rgb1
        assert rgb1.shape[-1] == 3
        ### Edit here for iteratively saving images per frame ###
        # plt.imsave(f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/SKJ-SurRoL-Development/surrol/data/rgb_array1_{self.counter}.png', rgb1)
        # if self.STEREO:
        #     rgb2 = output.rgb2

        seg=np.array((seg==6 )| (seg==1)).astype(int)
        #seg=np.resize(seg,(320,240))
        
        #print('depth : ', np.max(depth))
        
        # seg = cv2.resize(seg, (320,240), interpolation =cv2.INTER_NEAREST)
        # depth = cv2.resize(depth, (320,240), interpolation =cv2.INTER_NEAREST)
        
        #plt.imsave('/home/student/code/regress_data8/seg/seg_{}.png'.format(self.counter),seg)
        # np.save('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/test/seg_npy/seg_{}.npy'.format(self.counter),seg)
        # np.save('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/test/depth/depth_{}.npy'.format(self.counter),depth)
        # cv2.imwrite('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/test/img/img_{}.png'.format(self.counter),cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
        
        self.img_list[self.counter]={}
        self.img_list[self.counter]['obs']=obs['observation']
        
        #self.img_list[self.counter]=obs['observation']

        # if self.counter>200:
        #     with open('/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/img_obs.pkl',"wb") as f:
        #     # with open('/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/SKJ-SurRoL-Development/surrol/data/img_obs.pkl',"wb") as f:
        #         pickle.dump(self.img_list,f)
        #     exit()
            # return
        
        return obs

    # 动作推断
    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy. I have moved the image collection function here. In the policy testing phase, we will only need the image and write a new function for action inference.
        """
        robot_joint_state = self.psm1.get_current_joint_position()
        # print(f"joint:,{robot_joint_state}")
        output = self.ecm.render_image(stereo=self.STEREO)
        mask=output.mask1
        rgb1=output.rgb1
        assert rgb1.shape[-1] == 3
        ### Edit here for iteratively saving images per frame ###
        # plt.imsave(f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/SKJ-SurRoL-Development/surrol/data/rgb_array1_{self.counter}.png', rgb1)
        if self.STEREO:
            rgb2 = output.rgb2
            # plt.imsave(f'/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/SKJ-SurRoL-Development/surrol/data/rgb_array2_{self.counter}.png', rgb2)
        # four waypoints executed in sequential order
        
        self.counter += 1
        print("CNT:", self.counter)
        action = np.zeros(5)
        action[4] = -0.5

        # sky edit
        waypoint = self._waypoints[0]       
        print("self._waypoints", self._waypoints)
        print("waypoint is:", waypoint)
        # if waypoint is None:
        #     print("aeiou")
        #     continue
            
        delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
        delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
        if np.abs(delta_pos).max() > 1:
            delta_pos /= np.abs(delta_pos).max()
        scale_factor = 0.4
        delta_pos *= scale_factor
        # 根据偏移量和航点状态生成动作
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            # 检查当前位置是否已经足够接近目标航点
        if not (np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2):
            print("hello world")
            self.image_list.append([np.array(rgb1), np.array(rgb2)])
            self.actions_list.append(action)

            self.qpos_list.append(robot_joint_state)

            if self.counter > 2:
                joint_diffs = [robot_joint_state[i]-self.prev_qpos[i]
                            for i in range(len(robot_joint_state))]
                # print("ddddddd:",joint_diffs)
                
                self.new_actions.append(joint_diffs)
        else:
            print("OMGOMGOMGOMGOMG.")
            # return action
            # if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                # print("OMGOMGOMGOMGOMG.")
                # self._waypoints[i] = None
            # break     

        # 原代码，使用注释掉上面的即可
        # for i, waypoint in enumerate(self._waypoints[:1]):
        #     print("self._waypoints", self._waypoints)
        #     print("waypoint is:", waypoint)
        #     if waypoint is None:
        #         print("aeiou")
        #         continue
                
        #     delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
        #     delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
        #     if np.abs(delta_pos).max() > 1:
        #         delta_pos /= np.abs(delta_pos).max()
        #     scale_factor = 0.4
        #     delta_pos *= scale_factor
        #     # 根据偏移量和航点状态生成动作
        #     action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
        #     # 检查当前位置是否已经足够接近目标航点
        #     if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
        #         print("OMGOMGOMGOMGOMG.")
        #         self._waypoints[i] = None
        #     break       
            # self._waypoints[i] = None
        if self.counter > 2:
            self.image_list.append([np.array(rgb1), np.array(rgb2)])
            self.actions_list.append(action)

            self.qpos_list.append(robot_joint_state)

            joint_diffs = [robot_joint_state[i]-self.prev_qpos[i]
                        for i in range(len(robot_joint_state))]
            # print("ddddddd:",joint_diffs)
            
            self.new_actions.append(joint_diffs)
        self.prev_qpos = robot_joint_state
        # sky
        # if self.counter > 1000:
        #     # image_action_pairs = {'images': self.image_list, 'actions': self.actions_list}
        #     image_action_qpos = {'images': self.image_list, 'actions': self.actions_list, 'qpos': robot_joint_state}
        #     # with open('/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/SKJ-SurRoL-Development/surrol/data/image_action_pairs.pkl',"wb") as f:
        #     # with open('/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/image_action_pairs.pkl',"wb") as f:
        #     with open('/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/image_action_qpos',"wb") as f:
        #         pickle.dump(image_action_qpos,f)
        #     exit()
        # Save data every 1000 samples
        if self.counter % 1000 == 0:
            self.save_data()

        return action
    
    def save_data(self):
        """
        Save the collected data to a pickle file.
        """
        data = {
            'images': self.image_list,
            'actions': self.actions_list,  # TODO 
            # 'actions': self.new_actions,
            'qpos': self.qpos_list
        }
        # save_path = f'/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/image_action_qpos_{self.episode_idx}.pkl'
        save_path = f'/root/autodl-tmp/IROS_SurRoL/rl/act-main-3/data/1921exp3-2/image_action_qpos_{self.episode_idx}.pkl'

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved data to {save_path}")
        self.episode_idx += 1

        # Clear the collected data for the next batch
        self.image_list = []
        self.actions_list = []
        self.qpos_list = []
        self.new_actions = []
        # Exit after saving 5 files
        if self.episode_idx >= 10:
            print("Data collection complete. Exiting.")
            exit()



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