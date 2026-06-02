import os
import time
import numpy as np

import pybullet as p
from Simulator.surrol.tasks.psm_env_rpy import PsmsEnv, goal_distance
from Simulator.surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera, 
    wrap_angle
)
import cv2
from Simulator.surrol.tasks.ecm_env import EcmEnv, goal_distance

from Simulator.surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from Simulator.surrol.const import ASSET_DIR_PATH
from Simulator.surrol.robots.ecm import Ecm

import collections 

class BiPegTransfer(PsmsEnv):
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))
    POSE_WALL = ((0.55, -0.1, 0.8), (0, 0, 0))
    WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0., 0.05), (0.686, 0.745))
    WORKSPACE_LIMITS2 = ((0.50, 0.60), (-0.05, 0.), (0.686, 0.745))
    SCALING = 5.
    QPOS_ECM = (0, 0.0, 0.1, 0)#(0, 0.5, 0.15, 0)
    ACTION_ECM_SIZE=3
    #for haptic device demo  
    haptic=True
    def __init__(self, render_mode=None, cid = -1, action_mode = 'yaw'):
        super(BiPegTransfer, self).__init__(render_mode, cid, action_mode)
        print("begin bi-pegtransfer")
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def _env_setup(self):
        super(BiPegTransfer, self)._env_setup()
        self.has_object = True

        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
                         target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
        self.ecm = Ecm((-0.05, 0.0, 0.8524),#(0.08, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)
        self.STEREO = True

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        workspace_limits = self.workspace_limits2
        pos = (workspace_limits[0][0],
               workspace_limits[1][0],
               workspace_limits[2][1])
        joint_positions = self.psm2.inverse_kinematics((pos, orn), self.psm2.EEF_LINK_INDEX)
        self.psm2.reset_joint(joint_positions)
        self.block_gripper = False

        # peg board
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'peg_board/peg_board.urdf'),
                            np.array(self.POSE_BOARD[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        self._pegs = np.arange(12)
        # np.random.shuffle(self._pegs[:6])
        # np.random.shuffle(self._pegs[6: 12])

        # blocks
        num_blocks = 1
        for i in self._pegs[6: 6 + num_blocks]:
            pos, orn = get_link_pose(self.obj_ids['fixed'][1], i)
            yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
            # print("abcd:",np.array(pos) + np.array([0, 0, 0.03]))
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'block/block.urdf'),
                                np.array(pos) + np.array([0, 0, 0.03]),
                                p.getQuaternionFromEuler((0, 0, yaw)),
                                useFixedBase=False,
                                globalScaling=self.SCALING)
            self.obj_ids['rigid'].append(obj_id)
        self._blocks = np.array(self.obj_ids['rigid'][-num_blocks:])
        # np.random.shuffle(self._blocks)
        for obj_id in self._blocks[:1]:
            # change color to red
            p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
            self.target_id = obj_id
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            # p.resetBasePositionAndOrientation(obj_id, pos, (0, 0, 0, 1))  # reduce difficulty
        self.obj_id, self.obj_link1, self.obj_link2 = self._blocks[0], 1, 2

        # obstacle
        # pos_1, _ = get_link_pose(self.obj_ids['fixed'][1], int(self._pegs[0]))
        # wall_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'obstacle/thinwall.urdf'),
        #                     np.array(pos_1) + np.array([0, -0.04, 0]),
        #                     p.getQuaternionFromEuler(self.POSE_WALL[1]),
        #                     useFixedBase=True,
        #                     globalScaling=self.SCALING)

    # def _set_action(self, action: np.ndarray):
    #     # simplified to a hand and drop by performing the first three steps
    #     obs = self._get_obs()
    #     if not self._waypoints_done[3]:  # 1: approach, 2: pick, 3: lift
    #         action = self.get_oracle_action(obs)
    #     super(BiPegTransfer, self)._set_action(action)

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        # TODO: may need to tune parameters
        # result = np.logical_and(
        #     goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
        #     np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 4e-3 * self.SCALING,
        # ).astype(np.float32)
        result = np.logical_and(np.logical_and(
            goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
            np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 4e-3 * self.SCALING,
        ).astype(np.float32), self._waypoints_done[-1] == True)
        # print("result: ", result)
        return result

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        color = [1, 0, 0, 1]  # Red color with full opacity
        p.changeVisualShape(self.obj_ids['fixed'][1], self._pegs[2], rgbaColor=[1, 0, 0, 1])   
        goal = np.array(get_link_pose(self.obj_ids['fixed'][1], self._pegs[2])[0])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = []  # eleven waypoints
        pos_obj1, orn_obj1 = get_link_pose(self.obj_id, self.obj_link1)
        pos_obj2, orn_obj2 = get_link_pose(self.obj_id, self.obj_link2)
        orn1 = p.getEulerFromQuaternion(orn_obj1)
        orn2 = p.getEulerFromQuaternion(orn_obj2)
        orn_eef1 = p.getEulerFromQuaternion(get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1])
        orn_eef2 = p.getEulerFromQuaternion(get_link_pose(self.psm2.body, self.psm2.EEF_LINK_INDEX)[1])
        
        roll1 =  orn1[0] - np.deg2rad(90)
        roll2 =  orn2[0] - np.deg2rad(90)

        pitch1 = orn1[1] if abs(wrap_angle(orn1[1] - orn_eef1[1])) < abs(wrap_angle(orn1[1] + np.pi - orn_eef1[1])) \
            else wrap_angle(orn1[1] + np.pi) 
        pitch2 = orn2[1] if abs(wrap_angle(orn2[1] - orn_eef2[1])) < abs(wrap_angle(orn2[1] + np.pi - orn_eef2[1])) \
            else wrap_angle(orn2[1] + np.pi) 

        yaw1 = orn1[2] if abs(wrap_angle(orn1[2] - orn_eef1[2])) < abs(wrap_angle(orn1[2] + np.pi - orn_eef1[2])) \
            else wrap_angle(orn1[2] + np.pi)  # minimize the delta yaw
        yaw2 = orn2[2] if abs(wrap_angle(orn2[2] - orn_eef2[2])) < abs(wrap_angle(orn2[2] + np.pi - orn_eef2[2])) \
            else wrap_angle(orn2[2] + np.pi)  # minimize the delta yaw

        # the corresponding peg position
        # pos_peg = get_link_pose(self.obj_ids['fixed'][1], self.obj_id - np.min(self._blocks) + 6)[0]  # 6 pegs
        pos_peg = get_link_pose(self.obj_ids['fixed'][1],
                                self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]  # 6 pegs

        pos_mid1 = [pos_obj1[0],
                    0. + pos_obj1[1] - pos_peg[1], pos_obj1[2] + 0.043 * self.SCALING]  # consider offset
        pos_mid2 = [pos_obj2[0],
                    0. + pos_obj2[1] - pos_peg[1], pos_obj2[2] + 0.043 * self.SCALING]  # consider offset
        
        rot_mat2 = np.array(p.getMatrixFromQuaternion(orn_obj2)).reshape(3, 3)
        grasp_offset = np.array([-0.006, 0, 0])
        pos_obj2 = pos_obj2 + np.dot(rot_mat2, grasp_offset)

        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1],
                                         pos_mid1[2] + 0.01 * self.SCALING, roll1, pitch1, yaw1, 0.5,

                                         pos_obj2[0], pos_obj2[1],
                                         pos_mid2[2], roll2, pitch2, yaw2, 0.5]))  # above object
        
        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1],
                                         pos_mid1[2] + 0.01 * self.SCALING, roll1, pitch1, yaw1, 0.5,

                                         pos_obj2[0], pos_obj2[1],
                                         pos_mid2[2]-0.02 * self.SCALING, roll2, pitch2, yaw2, 0.5]))  # above object

        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1],
                                         pos_mid1[2] + 0.01 * self.SCALING, roll1, pitch1, yaw1, 0.5,

                                         pos_obj2[0], pos_obj2[1],
                                         pos_obj2[2] + (0.003 + 0.0102) * self.SCALING, roll2, pitch2, yaw2, 0.5]))  # approach

        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1],
                                         pos_mid1[2] + 0.01 * self.SCALING, roll1, pitch1, yaw1, 0.5,

                                         pos_obj2[0], pos_obj2[1],
                                         pos_obj2[2] + (0.003 + 0.0102) * self.SCALING, roll2, pitch2, yaw2, -0.5]))  # psm2 grasp

        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1],
                                         pos_mid1[2] + 0.01 * self.SCALING, roll1, pitch1, yaw1, 0.5,
                                         pos_obj2[0], pos_obj2[1],
                                         pos_mid2[2]+0.01, roll2, pitch2, yaw2, -0.5]))  # lift up

        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1], pos_mid1[2] + 0.01 * self.SCALING, roll1, pitch1, yaw1, 0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2]+0.01, roll2, pitch2,yaw2, -0.5]))  # move to middle

        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1], pos_mid1[2], roll1, pitch1,yaw1, 0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2]+0.01,  roll2, pitch2,yaw2, -0.5]))  # psm1 pre grasp

        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1], pos_mid1[2], roll1, pitch1,yaw1, -0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2]+0.01,  roll2, pitch2,yaw2, -0.5]))  # psm1 grasp

        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1], pos_mid1[2], roll1, pitch1,yaw1, -0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2],  roll2, pitch2,yaw2, 0.5]))  # psm2 release
                                         
        self._waypoints.append(np.array([pos_mid1[0], pos_mid1[1], pos_mid1[2], roll1, pitch1,yaw1, -0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2] + 0.01 * self.SCALING,  roll2, pitch2,yaw2, 0.5]))  # psm2 lift up

        pos_place = [self.goal[0] + pos_obj1[0] - pos_peg[0],
                     self.goal[1] + pos_obj1[1] - pos_peg[1], pos_mid1[2]]  # consider offset
        self._waypoints.append(np.array([pos_place[0], pos_place[1], pos_place[2], roll1, pitch1,yaw1, -0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2] + 0.01 * self.SCALING, roll2, pitch2,
                                         yaw2, 0.5]))  # above goal
        self._waypoints.append(np.array([pos_place[0], pos_place[1], pos_place[2]-0.06, roll1, pitch1,yaw1, -0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2] + 0.01 * self.SCALING,roll2, pitch2,
                                         yaw2, 0.5]))  # release to goal
        self._waypoints.append(np.array([pos_place[0], pos_place[1], pos_place[2]-0.06, roll1, pitch1,yaw1, 0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2] + 0.01 * self.SCALING,roll2, pitch2,
                                         yaw2, 0.5]))  # release to goal
        self._waypoints.append(np.array([pos_place[0], pos_place[1], pos_place[2] + 0.02,roll1, pitch1,yaw1, 0.5,
                                         pos_mid2[0], pos_mid2[1], pos_mid2[2] + 0.01 * self.SCALING,roll2, pitch2,
                                         yaw2, 0.5]))  # lift psm
        self._waypoints_done = [False] * len(self._waypoints)

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self.haptic is True:
            # print(f'meet due to hardcoe')
            return True
        else:
            pose = get_link_pose(self.obj_id, -1)
            return pose[0][2] > self.goal[2] + 0.01 * self.SCALING  # reduce difficulty

    # sky
    # 获取机器人的当前状态、目标位置和相机图像，打包成字典obs返回
    def project_to_pixel(self, centroid, img_shape):
        """
        Convert camera coordinates to pixel coordinates.
        
        :param pos_cam: Position in camera coordinates (NDC), expected shape (3, 1) or (4, 1).
        :param img_shape: Shape of the image (height, width).
        :return: Pixel coordinates (x, y).
        """
        height, width = img_shape
        
        # Extract normalized device coordinates
        x_ndc = centroid[0]
        y_ndc = centroid[1]

        # Convert NDC to pixel coordinates
        x_pixel = int((x_ndc + 1) * (width / 2))
        y_pixel = int((y_ndc + 1) * (height / 2))  # Invert y-direction

        # Clip to ensure within bounds
        x_pixel = np.clip(x_pixel, 0, width - 1)
        y_pixel = np.clip(y_pixel, 0, height - 1)
        
        return np.array([x_pixel, y_pixel])
    
    def _get_obs(self) -> dict:
        psm1_state = self._get_robot_state(0)
        psm2_state = self._get_robot_state(1)
        robot_state = np.concatenate([psm1_state, psm2_state])
        robot_joint_state_1 = self.psm1.get_current_joint_position()
        robot_joint_state_2 = self.psm2.get_current_joint_position()       
        # may need to modify
        if self.has_object:
            pos, _ = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
            # waypoint1
            pos, orn = get_link_pose(self.obj_id, self.obj_link1)
            waypoint_pos1 = np.array(pos)
            waypoint_rot1 = np.array(p.getEulerFromQuaternion(orn))
            # waypoint2
            pos, orn = get_link_pose(self.obj_id, self.obj_link2)
            waypoint_pos2 = np.array(pos)
            waypoint_rot2 = np.array(p.getEulerFromQuaternion(orn))
            # gripper state
            object_rel_pos1 = object_pos - robot_state[0: 3]
            object_rel_pos2 = object_pos - robot_state[7: 10]
        else:
            object_pos = waypoint_pos1 = waypoint_rot1 = waypoint_pos2 = waypoint_rot2 = \
                object_rel_pos1 = object_rel_pos2 = np.zeros(0)

        if self.has_object:
            achieved_goal = object_pos.copy() if not self._waypoint_goal else waypoint_pos1.copy()
        else:
            # tip position
            pos1 = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
            pos2 = np.array(get_link_pose(self.psm2.body, self.psm2.TIP_LINK_INDEX)[0])
            achieved_goal = np.concatenate([pos1, pos2])

        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos1.ravel(), object_rel_pos2.ravel(),
            waypoint_pos1.ravel(), waypoint_rot1.ravel(),
            waypoint_pos2.ravel(), waypoint_rot2.ravel()  # achieved_goal.copy(),
        ])

        """
            Robot state shape: (7,)
            Object position shape (flattened): (3,)
            Object relative position shape (flattened): (3,)
            Waypoint position shape (flattened): (3,)
            Waypoint rotation shape (flattened): (3,)
            Robot joint state shape: 6
            Observation shape: (25,)
        """
        # obs = {
        #     'observation': observation.copy(),
        #     'achieved_goal': achieved_goal.copy(),
        #     'desired_goal': self.goal.copy(),
        # }
        # Render the images (using the ECM)
        output = self.ecm.render_image(stereo=self.STEREO, scaling=self.SCALING)
        seg = output.mask1
        rgb1 = output.rgb1
        depth = output.depth1

        obs = collections.OrderedDict()

        obs['observation'] = observation.copy()
        obs['achieved_goal'] = achieved_goal.copy()
        obs['desired_goal'] = self.goal.copy()

        # obs['qpos'] = robot_joint_state_1 + robot_joint_state_2
        # obs['env_state'] = robot_state
        obs['qpos'] = robot_state
        # print("new qpos: ", robot_state)

        obs['images'] = dict()
        obs['images']['rgb1']= rgb1
        if self.STEREO:
            rgb2 = output.rgb2
            obs['images']['rgb2']= rgb2
        # sky
        obs['images']['mask'] = seg
        obs['images']['depth'] = depth

        tip1_pos, _ = get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)
        tip2_pos, _ = get_link_pose(self.psm2.body, self.psm2.TIP_LINK_INDEX)
        block_pos, _ = get_link_pose(self.obj_id, -1)
        tip1, tip2, block, tip1_pixel, tip2_pixel, block_pixel = self.save_cropped_tips_and_block(rgb1, tip1_pos, tip2_pos, block_pos, rgb1.shape[:2], crop_size=(64,64), prefix='oracle')

        obs['images']['foveal_block'] = block
        # obs['images']['foveal_tip1'] = tip1
        # obs['images']['foveal_tip2'] = tip2
        return obs

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # eleven waypoints executed in sequential order

        # sky
        # robot_joint_state = self.psm1.get_current_joint_position() + self.psm2.get_current_joint_position()
        robot_state = np.concatenate([self._get_robot_state(0), self._get_robot_state(1)]) 
        # print("BI ROBOT_STATE: ", robot_state)
        output = self.ecm.render_image(stereo=self.STEREO, scaling=self.SCALING)
        mask_ori=output.mask1
        mask_no_arm = np.array((mask_ori==self.target_id))
        mask_target = np.array((mask_ori==1)|(mask_ori==4)|(mask_ori==self.target_id))
        rgb1 = output.rgb1
        depth = output.depth1

        # Save cropped images around both tips and the block
        tip1_pos, _ = get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)
        tip2_pos, _ = get_link_pose(self.psm2.body, self.psm2.TIP_LINK_INDEX)
        block_pos, _ = get_link_pose(self.obj_id, -1)
        tip1, tip2, block, tip1_pixel, tip2_pixel, block_pixel = self.save_cropped_tips_and_block(rgb1, tip1_pos, tip2_pos, block_pos, rgb1.shape[:2], crop_size=(64,64), prefix='oracle')

        assert rgb1.shape[-1] == 3
        if self.STEREO:
            rgb2 = output.rgb2

        action = np.zeros(14)
        for i, waypoint in enumerate(self._waypoints):
            if self._waypoints_done[i]:
                continue
            delta_pos1 = (waypoint[0: 3] - obs['observation'][0: 3]) / 0.01 / self.SCALING    # psm1
            delta_roll1 = (waypoint[3] - obs['observation'][3]) 
            delta_pitch1 = (waypoint[4] - obs['observation'][4])
            delta_yaw1 = (waypoint[5] - obs['observation'][5]).clip(-1, 1)

            delta_pos2 = (waypoint[7: 10] - obs['observation'][7: 10]) / 0.01 / self.SCALING   # psm2
            
            delta_roll2 = (waypoint[10] - obs['observation'][10]) 
            delta_pitch2 = (waypoint[11] - obs['observation'][11]) 
            delta_yaw2 = (waypoint[12] - obs['observation'][12]).clip(-1, 1)
            if np.abs(delta_pos1).max() > 1:
                delta_pos1 /= np.abs(delta_pos1).max()
            if np.abs(delta_pos2).max() > 1:
                delta_pos2 /= np.abs(delta_pos2).max()
            scale_factor = 0.3
            delta_pos1 *= scale_factor
            delta_pos2 *= scale_factor
            action = np.array([delta_pos1[0], delta_pos1[1], delta_pos1[2], delta_roll1, delta_pitch1, delta_yaw1, waypoint[6],
                               delta_pos2[0], delta_pos2[1], delta_pos2[2], delta_roll2, delta_pitch2, delta_yaw2, waypoint[13]])  # waypoint[4,9]
            # print(' dis: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            #     np.linalg.norm(delta_pos1), np.abs(delta_yaw1),
            #     np.linalg.norm(delta_pos2), np.abs(delta_yaw2)))


            if np.linalg.norm(delta_pos1) * 0.01 / scale_factor < 2e-3 \
                    and np.linalg.norm(delta_pos2) * 0.01 / scale_factor < 2e-3:
                self._waypoints_done[i] = True
            break

        return action, rgb1, rgb2, mask_ori, mask_no_arm, mask_target, depth, robot_state, i, tip1, tip2, block, [tip1_pixel, tip2_pixel, block_pixel]
    
    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])
    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)

    def save_cropped_tips_and_block(self, rgb, tip1_pos, tip2_pos, block_pos, img_shape, crop_size, save_dir='cropped_images', prefix='img'):
        """
        Save cropped images around both arm tips and the block by projecting their 3D positions to pixel coordinates.
        Args:
            rgb: RGB image (H, W, 3)
            tip1_pos: 3D position of the first tool tip
            tip2_pos: 3D position of the second tool tip
            block_pos: 3D position of the block
            img_shape: (height, width) of the image
            crop_size: int, size of the square crop
            save_dir: str, directory to save images
            prefix: str, prefix for saved filenames
        """
        import os
        import cv2
        os.makedirs(save_dir, exist_ok=True)
        
        def crop_around_pixel(center, name):
            x_c, y_c = center
            halfx = crop_size[1] // 2
            halfy = crop_size[0] // 2
            x1, x2 = max(0, x_c - halfx), min(rgb.shape[1], x_c + halfx)
            y1, y2 = max(0, y_c - halfy), min(rgb.shape[0], y_c + halfy)
            crop = rgb[y1:y2, x1:x2]
            # print(f'Crop {name} image at pixel ({x_c}, {y_c}) with shape {crop.shape}')
            # filename = os.path.join(save_dir, f'{prefix}_{name}.png')
            # cv2.imwrite(filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            # print(f'Saved cropped {name} image to {filename}')
            return crop
        # Project positions to pixel coordinates
        tip1_pixel = self.project_to_pixel(self.ecm.get_centroid_proj(tip1_pos), img_shape)
        tip2_pixel = self.project_to_pixel(self.ecm.get_centroid_proj(tip2_pos), img_shape)
        block_pixel = self.project_to_pixel(self.ecm.get_centroid_proj(block_pos), img_shape)
        tip1 = crop_around_pixel(tip1_pixel, 'tip1')
        tip2 = crop_around_pixel(tip2_pixel, 'tip2')
        block = crop_around_pixel(block_pixel, 'block')
        return tip1, tip2, block, tip1_pixel, tip2_pixel, block_pixel


if __name__ == "__main__":
    env = BiPegTransfer(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)