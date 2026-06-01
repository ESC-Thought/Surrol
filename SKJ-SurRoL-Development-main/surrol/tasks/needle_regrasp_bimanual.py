import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env_rpy import PsmsEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera, 
    wrap_angle,
    step
)
from surrol.utils.robotics import get_matrix_from_pose_2d
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm

import collections
class NeedleRegrasp(PsmsEnv):
    # ACTION_MODE = 'pitch'
    WORKSPACE_LIMITS1 = ((0.55, 0.6), (0.01, 0.08), (0.695, 0.745))
    WORKSPACE_LIMITS2 = ((0.55, 0.6), (-0.08, -0.01), (0.695, 0.745))
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.075, 0)    # (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3

    def __init__(self, render_mode=None, cid = -1, action_mode = 'yaw'):
        super(NeedleRegrasp, self).__init__(render_mode, cid, action_mode)
        print("bigin needle regrasp")
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def _env_setup(self):
        super(NeedleRegrasp, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True

        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        # self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
        #                scaling=self.SCALING)
        self.ecm = Ecm((0.0865, 0.0, 0.7694), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)
        self.STEREO = True

        # robot
        for psm, workspace_limits in ((self.psm1, self.workspace_limits1), (self.psm2, self.workspace_limits2)):
            pos = (workspace_limits[0].mean(),
                   workspace_limits[1].mean(),
                   workspace_limits[2].mean())
            # orn = p.getQuaternionFromEuler(np.deg2rad([0, np.random.uniform(-45, -135), -90]))
            orn = p.getQuaternionFromEuler(np.deg2rad([0, -90, -90]))  # reduce difficulty

            # psm.reset_joint(self.QPOS_PSM1)
            joint_positions = psm.inverse_kinematics((pos, orn), psm.EEF_LINK_INDEX)
            psm.reset_joint(joint_positions)

        self.block_gripper = False  # set the constraint
        psm = self.psm1
        workspace_limits = self.workspace_limits1

        # needle
        limits_span = (workspace_limits[:, 1] - workspace_limits[:, 0]) / 3
        sample_space = workspace_limits.copy()
        sample_space[:, 0] += limits_span
        sample_space[:, 1] -= limits_span
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm_RL.urdf'),
                            (0.01 * self.SCALING, 0, 0),
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1, self.obj_link2 = self.obj_ids['rigid'][0], 4, 5

        while True:
            # open the jaw
            psm.open_jaw()
            # TODO: strange thing that if we use --num_env=1 with openai baselines, the qs vary before and after step!
            step(0.5)

            # set the position until the psm can grasp it
            pos_needle = np.random.uniform(low=sample_space[:, 0], high=sample_space[:, 1])
            pitch = np.random.uniform(low=-105., high=-75.)  # reduce difficulty
            orn_needle = p.getQuaternionFromEuler(np.deg2rad([-90, pitch, 90]))
            p.resetBasePositionAndOrientation(obj_id, pos_needle, orn_needle)

            # record the needle pose and move the psm to grasp the needle
            pos_waypoint, orn_waypoint = get_link_pose(obj_id, self.obj_link2)  # the right side waypoint
            orn_waypoint = np.rad2deg(p.getEulerFromQuaternion(orn_waypoint))
            p.resetBasePositionAndOrientation(obj_id, (0, 0, 0.01 * self.SCALING), (0, 0, 0, 1))

            # get the eef pose according to the needle pose
            orn_tip = p.getQuaternionFromEuler(np.deg2rad([90, -90 - orn_waypoint[1], 90]))
            pose_tip = [pos_waypoint + np.array([0.0015 * self.SCALING, 0, 0]), orn_tip]
            pose_eef = psm.pose_tip2eef(pose_tip)

            # move the psm
            pose_world = get_matrix_from_pose_2d(pose_eef)
            action_rcm = psm.pose_world2rcm(pose_world)
            success = psm.move(action_rcm)
            if success is False:
                continue
            step(1)
            p.resetBasePositionAndOrientation(obj_id, pos_needle, orn_needle)
            cid = p.createConstraint(obj_id, -1, -1, -1,
                                     p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos_needle,
                                     childFrameOrientation=orn_needle)
            psm.close_jaw()
            step(0.5)
            p.removeConstraint(cid)
            self._activate(0)
            self._step_callback()
            step(1)
            self._step_callback()
            if self._activated >= 0:
                break

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits2
        goal = workspace_limits.mean(axis=1) + np.random.randn(3) * 0.005 * self.SCALING
        goal.clip(workspace_limits[:, 0], workspace_limits[:, 1])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None, None]  # six waypoints
        # pos_obj1, _ = get_link_pose(self.obj_id, self.obj_link2)  # ?
        # pos_obj2, _ = get_link_pose(self.obj_id, self.obj_link1)
        pos_obj1, orn_obj1 = get_link_pose(self.obj_id, self.obj_link2)  # ?
        pos_obj2, orn_obj2 = get_link_pose(self.obj_id, self.obj_link1)
        orn1 = p.getEulerFromQuaternion(orn_obj1)
        orn2 = p.getEulerFromQuaternion(orn_obj2)

        pos_obj1, pos_obj2 = np.array(pos_obj1), np.array(pos_obj2)
        pos_dis = np.linalg.norm(pos_obj1 - pos_obj2)
        pitch1, pitch2 = np.deg2rad(-30), np.deg2rad(-30)
        jaw = 0.8

        pos_tip1 = (pos_obj1[0] + 0.002 * self.SCALING, pos_dis / 2, pos_obj1[2])
        orn_tip1 = p.getQuaternionFromEuler(np.deg2rad([90, -30, 90]))
        pose_tip1 = [pos_tip1, orn_tip1]
        # pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)  
        pos_eef1, orn_eef1 = self.psm1.pose_tip2eef(pose_tip1)  
        pos_tip2 = (pos_obj1[0] - 0.002 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        orn_tip2 = p.getQuaternionFromEuler(np.deg2rad([90, -150, 90]))
        pose_tip2 = [pos_tip2, orn_tip2]
        # pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        pos_eef2, orn_eef2 = self.psm2.pose_tip2eef(pose_tip2)

        # roll1 = orn1[0] - np.deg2rad(90)
        # roll2 = orn2[0] - np.deg2rad(90)
        roll1 =  np.deg2rad(0)
        roll2 =  - np.deg2rad(180)

        yaw1 =  - np.deg2rad(90)
        yaw2 =  np.deg2rad(90)
        # yaw1 = orn1[2] if abs(wrap_angle(orn1[2] - orn_eef1[2])) < abs(wrap_angle(orn1[2] + np.pi - orn_eef1[2])) \
        #     else wrap_angle(orn1[2] + np.pi)  # minimize the delta yaw
        # yaw2 = orn2[2] if abs(wrap_angle(orn2[2] - orn_eef2[2])) < abs(wrap_angle(orn2[2] + np.pi - orn_eef2[2])) \
        #     else wrap_angle(orn2[2] + np.pi)  # minimize the delta yaw
        # pitch1 = orn1[1] if abs(wrap_angle(orn1[1] - orn_eef1[1])) < abs(wrap_angle(orn1[1] + np.pi - orn_eef1[1])) \
        #     else wrap_angle(orn1[1] + np.pi) 
        # pitch2 = orn2[1] if abs(wrap_angle(orn2[1] - orn_eef2[1])) < abs(wrap_angle(orn2[1] + np.pi - orn_eef2[1])) \
        #     else wrap_angle(orn2[1] + np.pi) 

        self._waypoints[0] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], roll1, pitch1, yaw1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], roll2, pitch2, yaw2, jaw])  # move to the middle

        pose_tip1[0] = (pos_obj1[0], pos_dis / 2, pos_obj1[2])
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)
        pose_tip2[0] = (pos_obj1[0] + 0.002 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[1] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], roll1, pitch1, yaw1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], roll2, pitch2, yaw2, jaw])  # psm2 approach waypoint
        self._waypoints[2] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], roll1, pitch1, yaw1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], roll2, pitch2, yaw2, -jaw])  # psm2 grasp
        self._waypoints[3] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], roll1, pitch1, yaw1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], roll2, pitch2, yaw2, -jaw])  # psm1 release
        pose_tip1[0] = (pos_obj1[0] - 0.005 * self.SCALING, pos_dis / 2 + 0.01 * self.SCALING, pos_obj1[2])
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)
        pose_tip2[0] = (pos_obj1[0] + 0.005 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[4] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2],roll1, pitch1, yaw1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], roll2, pitch2, yaw2, -jaw])  # psm1 move middle
        pose_tip2[0] = (self.goal[0], self.goal[1], self.goal[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[5] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2],roll1, pitch1, yaw1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], roll2, pitch2, yaw2, -jaw])  # place

    def _meet_contact_constraint_requirement(self):
        """ add a contact constraint to the grasped needle to make it stable
        """
        return True

    # sky
    # 获取机器人的当前状态、目标位置和相机图像，打包成字典obs返回
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

        # Render the images (using the ECM)
        output = self.ecm.render_image(stereo=self.STEREO)
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

        assert rgb1.shape[-1] == 3
        return obs


    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        # sky
        # robot_joint_state = self.psm1.get_current_joint_position() + self.psm2.get_current_joint_position()
        robot_state = np.concatenate([self._get_robot_state(0), self._get_robot_state(1)]) 
        # print("BI ROBOT_STATE: ", robot_state)
        output = self.ecm.render_image(stereo=self.STEREO)
        mask_ori=output.mask1
        self.target_id = 3 # TODO
        mask_target = np.array((mask_ori==1)|(mask_ori==4)|(mask_ori==self.target_id))
        rgb1 = output.rgb1
        depth = output.depth1
        assert rgb1.shape[-1] == 3
        if self.STEREO:
            rgb2 = output.rgb2
            
        action = np.zeros(14)
        action[6], action[13] = 0.8, -0.8
        pitch_scaling = np.deg2rad(15)
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos1 = (waypoint[0: 3] - obs['observation'][0: 3]) / 0.01 / self.SCALING
            delta_roll1 = waypoint[3]            
            delta_pitch1 = ((waypoint[4] - obs['observation'][4]) / pitch_scaling).clip(-1, 1)
            delta_yaw1 = waypoint[5]

            delta_pos2 = (waypoint[7: 10] - obs['observation'][7: 10]) / 0.01 / self.SCALING
            delta_roll2 = waypoint[10]
            delta_pitch2 = ((waypoint[11] - obs['observation'][11]) / pitch_scaling).clip(-1, 1)
            delta_yaw2 = waypoint[12]
            if np.abs(delta_pos1).max() > 1:
                delta_pos1 /= np.abs(delta_pos1).max()
            if np.abs(delta_pos2).max() > 1:
                delta_pos2 /= np.abs(delta_pos2).max()
            scale_factor = 0.5
            delta_pos1 *= scale_factor
            delta_pos2 *= scale_factor
            action = np.array([delta_pos1[0], delta_pos1[1], delta_pos1[2], delta_roll1, delta_pitch1, delta_yaw1, waypoint[6],
                               delta_pos2[0], delta_pos2[1], delta_pos2[2], delta_roll2, delta_pitch2, delta_yaw2, waypoint[13]])
            if np.linalg.norm(delta_pos1) * 0.01 / scale_factor < 1e-4 and np.abs(delta_pitch1) < 2. \
                    and np.linalg.norm(delta_pos2) * 0.01 / scale_factor < 1e-4 and np.abs(delta_pitch2) < 2.:
                self._waypoints[i] = None
            break

        return action, rgb1, rgb2, mask_ori, mask_target, depth, robot_state, i
    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])
    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)




if __name__ == "__main__":
    env = NeedleRegrasp(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
