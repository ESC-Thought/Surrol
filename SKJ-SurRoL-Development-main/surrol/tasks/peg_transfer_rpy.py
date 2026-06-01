import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env_rpy import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm

# sky
import collections

class PegTransfer(PsmEnv):
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.686, 0.745))
    SCALING = 5.

    QPOS_ECM = (0, 0.7, 0.2, 0)
    ACTION_ECM_SIZE=3
    #for haptic device demo
    haptic=True

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, cid = -1, action_mode = 'yaw'):
        super(PegTransfer, self).__init__(render_mode, cid, action_mode)
        print("begin pegtransfer")
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )


    def _env_setup(self):
        super(PegTransfer, self)._env_setup()
        self.has_object = True

        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
                         target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)

        self.STEREO = True
        # self.ecm.reset_joint((3.3482885360717773, -0.0017351149581372738, 4.2447919845581055,0))
        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # peg board
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'peg_board/peg_board.urdf'),
                            np.array(self.POSE_BOARD[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        # print(f'peg transfer\' board size: {p.getVisualShapeData(obj_id)}')
        # group = 1#other objects don't collide with me
        # mask=1 # don't collide with any other object
        # p.setCollisionFilterGroupMask(obj_id, 0,group, mask)
        self._pegs = np.arange(12)
        np.random.shuffle(self._pegs[:6])
        np.random.shuffle(self._pegs[6: 12])
        # print(self._pegs)
        self._pegs = [2,1,0,3,4,5,6,7,9,11,10,8]
        self._cnt = 0
        # blocks
        num_blocks = 4
        # for i in range(6, 6 + num_blocks):
        for i in self._pegs[6: 6 + num_blocks]:
            pos, orn = get_link_pose(self.obj_ids['fixed'][1], i)
            yaw = (np.random.rand() - 0.5) * np.deg2rad(30)
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'block/block.urdf'),
                                np.array(pos) + np.array([0, 0, 0.03]),
                                p.getQuaternionFromEuler((0, 0, yaw)),
                                useFixedBase=False,
                                globalScaling=self.SCALING)
            # print(f"peg obj id: {obj_id}.")
            self.obj_ids['rigid'].append(obj_id)
        self._blocks = np.array(self.obj_ids['rigid'][-num_blocks:])
        # print(f'peg transfer\' peg size: {p.getVisualShapeData(obj_id)}')

        np.random.shuffle(self._blocks)
        for obj_id in self._blocks[:1]:
            # change color to red
            p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        self.obj_id, self.obj_link1 = self._blocks[0], 1

        # self._pegs = [2,1,0,3,4,5,6,7,9,11,10,8]
        # self._pegs = [3,1,4,5,6,8,0,2,7,9,10,11]
        # # blocks
        # num_blocks = 6
        # # for i in range(6, 6 + num_blocks):
        # for i in self._pegs[6: 6 + num_blocks]:
        #     pos, orn = get_link_pose(self.obj_ids['fixed'][1], i)
        #     yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
        #     obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'block/block_haptic.urdf'),
        #                         np.array(pos) + np.array([0, 0, 0.03]),
        #                         p.getQuaternionFromEuler((0, 0, yaw)),
        #                         useFixedBase=False,
        #                         globalScaling=self.SCALING)
        #     print(f"peg obj id: {obj_id}.")
        #     self.obj_ids['rigid'].append(obj_id)
        # self._blocks = np.array(self.obj_ids['rigid'][-num_blocks:])
        # # np.random.shuffle(self._blocks)
        # for obj_id in self._blocks[:3]:
        #     # change color to red
        #     p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        # self.obj_id, self.obj_link1 = self._blocks[2], -1
        # print(self.obj_ids['fixed'])
        # print(f'goal peg:{obj_id}')
    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        # TODO: may need to tune parameters
        result = np.logical_and(np.logical_and(
            goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
            np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 2e-2 * self.SCALING
        ).astype(np.float32), (self._waypoints[-1] == None)).all()
        # print("result: ", result)
        return result

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        ## TODO: if workspace limits are needed?
        # print("self.obj_ids[fixed]:", self.obj_ids['fixed'])
        # print("self.obj_ids[fixed][1]:", self.obj_ids['fixed'][1])
        # print("self._pegs:", self._pegs)     
        # print("self._pegs[0]:", self._pegs[0])
        color = [1, 0, 0, 1]  # Red color with full opacity
        p.changeVisualShape(self.obj_ids['fixed'][1], self._pegs[0], rgbaColor=[1, 0, 0, 1])     
#         self.obj_ids[fixed]: [3, 5]
#         self.obj_ids[fixed][1]: 5
#         self._pegs: [2, 1, 0, 3, 4, 5, 6, 7, 9, 11, 10, 8]
#       self._pegs[0]: 2
        goal = np.array(get_link_pose(self.obj_ids['fixed'][1], self._pegs[0])[0])
        print("GOAL:", goal)
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None, None, None, None]  # six waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        
        print(f"Current ORN: [1]{orn[0]}, [2]{orn[1]}")
        # sky
        # roll = orn[0] if abs(wrap_angle(orn[0] - orn_eef[0])) < abs(wrap_angle(orn[0] + np.pi - orn_eef[0])) \
        #     else wrap_angle(orn[0] + np.pi)
        # roll = np.deg2rad(-90)
        roll = orn[0] - np.deg2rad(90)
        pitch = orn[1] if abs(wrap_angle(orn[1] - orn_eef[1])) < abs(wrap_angle(orn[1] + np.pi - orn_eef[1])) \
            else wrap_angle(orn[1] + np.pi)
        # pitch = orn[1]
        # print("PITCH: ",pitch)
        # pitch = np.deg2rad(10)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw
        rot_mat = np.array(p.getMatrixFromQuaternion(orn_obj)).reshape(3, 3)
        grasp_offset = np.array([-0.006, 0, 0])
        pos_obj = pos_obj + np.dot(rot_mat, grasp_offset)
        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, roll, pitch, yaw, 0.5])  # above object
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.01) * self.SCALING, roll, pitch, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.01) * self.SCALING, roll, pitch, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, roll, pitch, yaw, -0.5])  # lift up

        # self._waypoints[0] = np.array([pos_obj[0]-0.0275, pos_obj[1]-0.005,
        #                                pos_obj[2] + 0.045 * self.SCALING, yaw, 0.5])  # above object
        # self._waypoints[1] = np.array([pos_obj[0]-0.0275, pos_obj[1]-0.005,
        #                                pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        # self._waypoints[2] = np.array([pos_obj[0]-0.0275, pos_obj[1]-0.005,
        #                                pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        # self._waypoints[3] = np.array([pos_obj[0]-0.0275, pos_obj[1]-0.005,
        #                                pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])  # lift up

        # pos_peg = get_link_pose(self.obj_ids['fixed'][1], self.obj_id - np.min(self._blocks) + 6)[0]  # 6 pegs
        pos_peg = get_link_pose(self.obj_ids['fixed'][1],
                                self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]  # 6 pegs
        pos_place = [self.goal[0] + pos_obj[0] - pos_peg[0],
                     self.goal[1] + pos_obj[1] - pos_peg[1], self._waypoints[0][2]]  # consider offset
        # pos_place = pos_place + np.dot(rot_mat, grasp_offset)
        self._waypoints[4] = np.array([pos_place[0], pos_place[1], pos_place[2], roll, pitch, yaw, -0.5])  # above goal
        self._waypoints[5] = np.array([pos_place[0], pos_place[1], pos_place[2]-0.04, roll, pitch, yaw, -0.5])  # release
        self._waypoints[6] = np.array([pos_place[0], pos_place[1], pos_place[2]-0.04, roll, pitch, yaw, 0.5])  # release
        self._waypoints[7] = np.array([pos_place[0], pos_place[1], pos_place[2]+0.02, roll, pitch, yaw, 0.5])  # release

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self.haptic is True:
            # print(f'meet due to hardcode')
            points_1 = p.getContactPoints(bodyA=self.psm1.body, linkIndexA=6)
            points_2 = p.getContactPoints(bodyA=self.psm1.body, linkIndexA=7)
            points_1 = [point[2] for point in points_1 if point[2] in self.obj_ids['rigid']]
            points_2 = [point[2] for point in points_2 if point[2] in self.obj_ids['rigid']]
            contact_List = list(set(points_1)&set(points_2))
            # print(f'joint contact item:{contact_List}')
            if len(contact_List)>0:
                return True
        else:
            pose = get_link_pose(self.obj_id, -1)
            # print(f'meet by checking distance')
            return pose[0][2] > self.goal[2] + 0.01 * self.SCALING
        return False

    # sky
    # 获取机器人的当前状态、目标位置和相机图像，打包成字典obs返回
    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        robot_joint_state = self.psm1.get_current_joint_position()
       
        pos, _ = get_link_pose(self.obj_id, -1)
        object_pos = np.array(pos)

        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)
        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        object_rel_pos = object_pos - robot_state[0: 3]
        
        # tip position
        # sky
        pos, _ = get_link_pose(self.obj_id, -1)
        achieved_goal = np.array(pos) if not self._waypoint_goal else waypoint_pos.copy()
        # print("ACHIEVED_GOAL: ", achieved_goal)
        # achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
            
        ### observation[-6:] = robot_joint_state is the robot joint state at this step ###
        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel(), robot_joint_state  # achieved_goal.copy(),
        ])

        # Render the images (using the ECM)
        output = self.ecm.render_image(stereo=self.STEREO)
        seg = output.mask1
        rgb1 = output.rgb1
        depth = output.depth1

        obs = collections.OrderedDict()

        obs['observation'] = observation.copy()
        obs['achieved_goal'] = achieved_goal.copy()
        obs['desired_goal'] = self.goal.copy()

        # sky
        obs['qpos'] = robot_joint_state
        obs['env_state'] = robot_state
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
        robot_joint_state = self.psm1.get_current_joint_position()
        output = self.ecm.render_image(stereo=self.STEREO)
        mask=output.mask1
        rgb1 = output.rgb1
        depth = output.depth1

        assert rgb1.shape[-1] == 3
        if self.STEREO:
            rgb2 = output.rgb2

        action = np.zeros(7)   # TODO
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            if i==4 or i==5:
                self._cnt+=1

            # Calculate the direction vector from current observation to the waypoint
            direction = waypoint[:3] - obs['observation'][:3]
            # Calculate the distance
            distance = np.linalg.norm(direction)
            # Normalize the direction vector


            if distance > 0.02:
                normalized_direction = direction / distance
                delta_pos = normalized_direction *0.03
            else:
                delta_pos = direction 

            # delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            
            # sky
            delta_roll = (waypoint[3] - obs['observation'][3])  #.clip(-0.4,0.4)
            delta_pitch = (waypoint[4] - obs['observation'][4])  #.clip(-0.4,0.4)
            #
            delta_yaw = (waypoint[5] - obs['observation'][5]).clip(-0.5, 0.5) # waypoint[3]
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.7   #  0.7
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_roll, delta_pitch, delta_yaw, waypoint[6]])  #waypoint[4]
            # if (i==4 and self._cnt>=25) or (np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(2.)):  # TODO
            #     self._cnt=0
            #     self._waypoints[i] = None
            # print("eeeeeeeee: ",np.abs(delta_yaw))
            # print("cnt: ", self._cnt)
            if i == 2:
                if self.psm1.get_current_jaw_position() != 0 and np.linalg.norm(delta_pos) *0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                   self._waypoints[i] = None              
            else:   
                if (i==4 and self._cnt>=25) or(i==5 and self._cnt>=25) or (np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < np.deg2rad(2.)):
                    self._cnt = 0
                    self._waypoints[i] = None
            break

        # return action
        # sky
        return action, rgb1, rgb2, mask, depth, robot_joint_state, i
        
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
    env = PegTransfer(render_mode='human')  # create one process and corresponding env

    # ?
    env.test()
    env.close()
    time.sleep(2)
