import os
import time
import collections

import numpy as np
import pybullet as p

from Simulator.surrol.const import ASSET_DIR_PATH
from Simulator.surrol.gym.surrol_goalenv import SurRoLGoalEnv
from Simulator.surrol.robots.ecm import Ecm
from Simulator.surrol.robots.psm import Psm2
from Simulator.surrol.utils.pybullet_utils import get_link_pose, reset_camera, wrap_angle
from Simulator.surrol.utils.robotics import get_euler_from_matrix, get_matrix_from_euler


BODY_ID_MASK = (1 << 24) - 1
LINK_ID_SHIFT = 24


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_float(value, default):
    if value is None:
        return float(default)
    return float(value)


def _parse_peg_indices(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = list(value)
    parsed = tuple(int(item) for item in items)
    return parsed if parsed else tuple(default)


class PegBlockPickPsm2(SurRoLGoalEnv):
    """PSM2-only peg block picking task with a 7D state/action interface.

    This task intentionally does not inherit from the bimanual BiPegTransfer
    environment. It recreates the peg board, block, ECM camera, and PSM2 control
    path directly so DP3 can train/evaluate with:

        qpos/action = [x, y, z, roll, pitch, yaw, jaw]
    """

    ACTION_SIZE = 7
    ACTION_MODE = "rpy"
    DISTANCE_THRESHOLD = 0.005

    POSE_PSM2 = ((0.05, -0.24, 0.8524), (0, 0, -(90 - 20) / 180 * np.pi))
    POSE_TABLE = ((0.5, 0, 0.001), (0, 0, 0))
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))
    WORKSPACE_LIMITS2 = ((0.50, 0.60), (-0.05, 0.0), (0.686, 0.745))

    SCALING = 5.0
    QPOS_ECM = (0, 0.0, 0.1, 0) # global and cropped_v2
    # QPOS_ECM = (0.0, np.deg2rad(50), 0.0, 0.0) # cropped_v3
    # QPOS_ECM = (a,b,c,d)，b增大->相机向上转动，c减小->视角变广

    DEFAULT_BLOCK_INITIAL_PEGS = (6,)
    DEFAULT_BLOCK_INITIAL_XY_JITTER = 0.0
    DEFAULT_WAYPOINT_POS_NOISE_STD = 0.01 * SCALING
    DEFAULT_WAYPOINT_POS_NOISE_CLIP = 0.03 * SCALING
    DEFAULT_WAYPOINT_NOISE_XY_ONLY = True
    DEFAULT_REQUIRE_GRASP_FOR_SUCCESS = False
    LIFT_SUCCESS_HEIGHT = 0.0175 * SCALING

    def __init__(
        self,
        render_mode=None,
        cid=-1,
        action_mode="rpy",
        randomize_block_initial_position=None,
        block_initial_peg_indices=None,
        block_initial_xy_jitter=None,
        waypoint_pos_noise_std=None,
        waypoint_pos_noise_clip=None,
        waypoint_noise_xy_only=None,
        require_grasp_for_success=None,
    ):
        if action_mode != "rpy":
            raise ValueError("PegBlockPickPsm2 currently supports only action_mode='rpy'.")

        if randomize_block_initial_position is None:
            randomize_block_initial_position = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_RANDOMIZE_BLOCK_INITIAL_POSITION",
                os.environ.get("SURROL_BIPEG_RANDOMIZE_BLOCK_INITIAL_POSITION"),
            )
        if block_initial_peg_indices is None:
            block_initial_peg_indices = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_BLOCK_INITIAL_PEGS",
                os.environ.get("SURROL_BIPEG_BLOCK_INITIAL_PEGS"),
            )
        if block_initial_xy_jitter is None:
            block_initial_xy_jitter = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_BLOCK_INITIAL_XY_JITTER",
                self.DEFAULT_BLOCK_INITIAL_XY_JITTER,
            )
        if waypoint_pos_noise_std is None:
            waypoint_pos_noise_std = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_WAYPOINT_POS_NOISE_STD",
                self.DEFAULT_WAYPOINT_POS_NOISE_STD,
            )
        if waypoint_pos_noise_clip is None:
            waypoint_pos_noise_clip = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_WAYPOINT_POS_NOISE_CLIP",
                self.DEFAULT_WAYPOINT_POS_NOISE_CLIP,
            )
        if waypoint_noise_xy_only is None:
            waypoint_noise_xy_only = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_WAYPOINT_NOISE_XY_ONLY",
                self.DEFAULT_WAYPOINT_NOISE_XY_ONLY,
            )
        if require_grasp_for_success is None:
            require_grasp_for_success = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_REQUIRE_GRASP_FOR_SUCCESS",
                self.DEFAULT_REQUIRE_GRASP_FOR_SUCCESS,
            )

        self.ACTION_MODE = action_mode
        self.randomize_block_initial_position = _parse_bool(
            randomize_block_initial_position,
            default=False,
        )
        self.block_initial_peg_indices = _parse_peg_indices(
            block_initial_peg_indices,
            self.DEFAULT_BLOCK_INITIAL_PEGS,
        )
        self.block_initial_xy_jitter = _parse_float(
            block_initial_xy_jitter,
            self.DEFAULT_BLOCK_INITIAL_XY_JITTER,
        )
        self.waypoint_pos_noise_std = _parse_float(
            waypoint_pos_noise_std,
            self.DEFAULT_WAYPOINT_POS_NOISE_STD,
        )
        self.waypoint_pos_noise_clip = _parse_float(
            waypoint_pos_noise_clip,
            self.DEFAULT_WAYPOINT_POS_NOISE_CLIP,
        )
        self.waypoint_noise_xy_only = _parse_bool(
            waypoint_noise_xy_only,
            default=self.DEFAULT_WAYPOINT_NOISE_XY_ONLY,
        )
        self.require_grasp_for_success = _parse_bool(
            require_grasp_for_success,
            default=self.DEFAULT_REQUIRE_GRASP_FOR_SUCCESS,
        )

        self.workspace_limits2 = (
            np.asarray(self.WORKSPACE_LIMITS2) + np.array([0.0, 0.0, 0.0102]).reshape(3, 1)
        ) * self.SCALING
        self.distance_threshold = self.DISTANCE_THRESHOLD * self.SCALING

        self.has_object = True
        self._waypoint_goal = False
        self._goal_plot = True
        self.block_gripper = False
        self._activated = -1
        self._contact_constraint = None
        self._contact_approx = False
        self.psm1 = None
        self.psm2 = None
        self.ecm = None
        self.STEREO = True

        self.block_source_peg_indices = []
        self.block_initial_xy_offsets = []
        self.initial_block_pos = None
        self.last_waypoint_pos_noise = np.zeros(3, dtype=np.float32)
        self.reset_waypoint = None
        self._waypoints = []
        self._waypoints_done = []

        super().__init__(render_mode=render_mode, cid=cid, action_mode=action_mode)
        print("begin peg-block-pick-psm2")

    @property
    def action_size(self):
        return self.ACTION_SIZE

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        return self._is_success(achieved_goal, desired_goal).astype(np.float32) - 1.0

    def _env_setup(self):
        self.obj_ids = {"fixed": [], "rigid": [], "deformable": []}
        self.has_object = True
        self._waypoint_goal = False
        self.block_gripper = False
        self._activated = -1
        self._contact_constraint = None
        self._contact_approx = False

        if self._render_mode == "human":
            reset_camera(
                yaw=90.0,
                pitch=-30.0,
                dist=0.82 * self.SCALING,
                target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING),
            )

        self.psm1 = None
        self.psm2 = Psm2(
            self.POSE_PSM2[0],
            p.getQuaternionFromEuler(self.POSE_PSM2[1]),
            scaling=self.SCALING,
        )
        self.psm2_eul = np.array(
            p.getEulerFromQuaternion(
                self.psm2.pose_rcm2world(self.psm2.get_current_position(), "tuple")[1]
            )
        )

        pos = (
            self.workspace_limits2[0][0],
            self.workspace_limits2[1][0],
            self.workspace_limits2[2][1],
        )
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm2.inverse_kinematics((pos, orn), self.psm2.EEF_LINK_INDEX)
        self.psm2.reset_joint(joint_positions)

        p.loadURDF(
            os.path.join(ASSET_DIR_PATH, "table/table.urdf"),
            np.array(self.POSE_TABLE[0]) * self.SCALING,
            p.getQuaternionFromEuler(self.POSE_TABLE[1]),
            globalScaling=self.SCALING,
        )

        goal_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, "sphere/sphere.urdf"),
            globalScaling=self.SCALING,
        )
        self.obj_ids["fixed"].append(goal_id)

        self.ecm = Ecm((-0.05, 0.0, 0.8524), scaling=self.SCALING) # global
        # self.ecm = Ecm((-0.08, -0.03, 0.8524), scaling=self.SCALING) # cropped_v2
        # self.ecm = Ecm((0.0, -0.03, 0.7), scaling=self.SCALING) # cropped_v3
        self.ecm.reset_joint(self.QPOS_ECM)
        # ECM((a,b,c), scaling=d)


        peg_board_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, "peg_board/peg_board.urdf"),
            np.array(self.POSE_BOARD[0]) * self.SCALING,
            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
            globalScaling=self.SCALING,
        )
        self.obj_ids["fixed"].append(peg_board_id)
        self._pegs = np.arange(12)

        num_blocks = 1
        self.block_source_peg_indices = []
        self.block_initial_xy_offsets = []
        peg_candidates = np.asarray(self.block_initial_peg_indices, dtype=np.int64)
        if self.randomize_block_initial_position:
            replace = peg_candidates.size < num_blocks
            block_source_pegs = np.random.choice(peg_candidates, size=num_blocks, replace=replace)
        else:
            block_source_pegs = peg_candidates[:num_blocks]

        for peg_index in block_source_pegs:
            peg_index = int(peg_index)
            self.block_source_peg_indices.append(peg_index)
            peg_pos, _ = get_link_pose(peg_board_id, peg_index)
            xy_offset = np.zeros(2, dtype=np.float32)
            if self.block_initial_xy_jitter > 0.0:
                xy_offset = np.random.uniform(
                    low=-self.block_initial_xy_jitter,
                    high=self.block_initial_xy_jitter,
                    size=2,
                ).astype(np.float32)
            self.block_initial_xy_offsets.append(xy_offset.copy())
            yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
            block_pos = np.asarray(peg_pos) + np.array([xy_offset[0], xy_offset[1], 0.03])
            block_id = p.loadURDF(
                os.path.join(ASSET_DIR_PATH, "block/block.urdf"),
                block_pos,
                p.getQuaternionFromEuler((0, 0, yaw)),
                useFixedBase=False,
                globalScaling=self.SCALING,
            )
            self.obj_ids["rigid"].append(block_id)

        self._blocks = np.asarray(self.obj_ids["rigid"][-num_blocks:])
        self.obj_id = int(self._blocks[0])
        self.obj_link1 = 1
        self.obj_link2 = 2
        self.target_id = self.obj_id
        p.changeVisualShape(self.obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))

        block_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
        self.initial_block_pos = np.asarray(block_pos, dtype=np.float32)

    def _get_robot_state(self) -> np.ndarray:
        pose_world = self.psm2.pose_rcm2world(self.psm2.get_current_position(), "tuple")
        jaw_angle = self.psm2.get_current_jaw_position()
        return np.concatenate(
            [
                np.asarray(pose_world[0]),
                np.asarray(p.getEulerFromQuaternion(pose_world[1])),
                np.asarray(jaw_angle).ravel(),
            ]
        ).astype(np.float32)

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state()
        object_pos, _ = get_link_pose(self.obj_id, -1)
        object_pos = np.asarray(object_pos, dtype=np.float32)
        waypoint_pos, waypoint_orn = get_link_pose(self.obj_id, self.obj_link2)
        waypoint_pos = np.asarray(waypoint_pos, dtype=np.float32)
        waypoint_rot = np.asarray(p.getEulerFromQuaternion(waypoint_orn), dtype=np.float32)
        object_rel_pos = object_pos - robot_state[:3]

        achieved_goal = object_pos.copy()
        observation = np.concatenate(
            [
                robot_state,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                waypoint_pos.ravel(),
                waypoint_rot.ravel(),
            ]
        ).astype(np.float32)

        rgb1, rgb2, mask, mask_no_arm, mask_target, depth = self._render_task_images()

        obs = collections.OrderedDict()
        obs["observation"] = observation.copy()
        obs["achieved_goal"] = achieved_goal.copy()
        obs["desired_goal"] = self.goal.copy()
        obs["qpos"] = robot_state.copy()
        obs["images"] = {
            "rgb1": rgb1,
            "rgb2": rgb2,
            "mask": mask,
            "mask_no_arm": mask_no_arm,
            "mask_target": mask_target,
            "depth": depth,
        }
        return obs

    def _set_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert len(action) == self.ACTION_SIZE, "PegBlockPickPsm2 expects a 7D action."
        action = action.copy()

        action[:3] *= 0.01 * self.SCALING
        pose_world = self.psm2.pose_rcm2world(self.psm2.get_current_position())
        pose_world[:3, 3] = np.clip(
            pose_world[:3, 3] + action[:3],
            self.workspace_limits2[:, 0] - [0.02, 0.02, 0.0],
            self.workspace_limits2[:, 1] + [0.02, 0.02, 0.08],
        )

        rot = get_euler_from_matrix(pose_world[:3, :3])
        action[3] *= np.deg2rad(20)
        roll = wrap_angle(rot[0] + action[3])
        action[4] *= np.deg2rad(20)
        pitch = np.clip(wrap_angle(rot[1] + action[4]), np.deg2rad(-90), np.deg2rad(90))
        action[5] *= np.deg2rad(20)
        yaw = wrap_angle(rot[2] + action[5])
        pose_world[:3, :3] = get_matrix_from_euler((roll, pitch, yaw))

        action_rcm = self.psm2.pose_world2rcm(pose_world)
        self.psm2.move(action_rcm)

        if action[6] < 0:
            self.psm2.close_jaw()
            self._activate()
        else:
            self.psm2.move_jaw(np.deg2rad(40))
            self._release()

    def _step_callback(self):
        if self.block_gripper or not self.has_object or self._activated < 0:
            return

        if self._contact_constraint is None:
            if not self._meet_contact_constraint_requirement():
                return
            contact_id = self._get_gripper_contact_object_id()
            if contact_id is None:
                return

            body_pose = p.getLinkState(self.psm2.body, self.psm2.EEF_LINK_INDEX)
            obj_pose = p.getBasePositionAndOrientation(contact_id)
            world_to_body = p.invertTransform(body_pose[0], body_pose[1])
            obj_to_body = p.multiplyTransforms(
                world_to_body[0],
                world_to_body[1],
                obj_pose[0],
                obj_pose[1],
            )
            self._contact_constraint = p.createConstraint(
                parentBodyUniqueId=self.psm2.body,
                parentLinkIndex=self.psm2.EEF_LINK_INDEX,
                childBodyUniqueId=contact_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=obj_to_body[0],
                parentFrameOrientation=obj_to_body[1],
                childFramePosition=(0, 0, 0),
                childFrameOrientation=(0, 0, 0),
            )
            p.changeConstraint(self._contact_constraint, maxForce=20)
        elif self._get_gripper_contact_object_id() is None and not self._contact_approx:
            self._release()

    def _activate(self):
        if self.block_gripper or self._activated >= 0:
            return

        if self._contact_approx:
            tip_pos, _ = get_link_pose(self.psm2.body, self.psm2.TIP_LINK_INDEX)
            obj_pos, _ = get_link_pose(self.obj_id, -1)
            if np.linalg.norm(np.asarray(tip_pos) - np.asarray(obj_pos)) < 2e-3 * self.SCALING:
                self._activated = 0
            return

        if self._get_gripper_contact_object_id() is not None:
            self._activated = 0

    def _release(self):
        if self.block_gripper or self._activated < 0:
            return

        self._activated = -1
        if self._contact_constraint is not None:
            try:
                p.changeConstraint(self._contact_constraint, maxForce=0)
                p.removeConstraint(self._contact_constraint)
            except Exception:
                pass
            self._contact_constraint = None

    def _get_gripper_contact_object_id(self):
        points_1 = p.getContactPoints(bodyA=self.psm2.body, linkIndexA=6)
        points_2 = p.getContactPoints(bodyA=self.psm2.body, linkIndexA=7)
        points_1 = [point[2] for point in points_1 if point[2] in self.obj_ids["rigid"]]
        points_2 = [point[2] for point in points_2 if point[2] in self.obj_ids["rigid"]]
        intersect = list(set(points_1) & set(points_2))
        return intersect[-1] if intersect else None

    def _meet_contact_constraint_requirement(self):
        return True

    def _is_success(self, achieved_goal, desired_goal):
        if self.initial_block_pos is None:
            return np.zeros_like(achieved_goal[..., -1], dtype=np.float32)

        block_pos, _ = get_link_pose(self.obj_id, -1)
        lifted = block_pos[2] > self.initial_block_pos[2] + self.LIFT_SUCCESS_HEIGHT
        if self.require_grasp_for_success:
            lifted = lifted and (self._activated >= 0 or self._contact_constraint is not None)
        return np.asarray(lifted, dtype=np.float32)

    def _sample_goal(self) -> np.ndarray:
        peg_board_id = self.obj_ids["fixed"][1]
        p.changeVisualShape(peg_board_id, int(self._pegs[2]), rgbaColor=[1, 0, 0, 1])
        goal = np.asarray(get_link_pose(peg_board_id, int(self._pegs[2]))[0], dtype=np.float32)
        return goal.copy()

    def _sample_goal_callback(self):
        if self.obj_ids["fixed"]:
            p.resetBasePositionAndOrientation(self.obj_ids["fixed"][0], self.goal, (0, 0, 0, 1))

        self._waypoints = self._build_waypoints()
        self._waypoints_done = [False] * len(self._waypoints)
        self.last_waypoint_pos_noise = self._sample_waypoint_pos_noise()
        self.reset_waypoint = self._waypoints[0].copy()
        self.reset_waypoint[:2] += self.last_waypoint_pos_noise[:2]
        self._reset_psm_to_waypoint(self.reset_waypoint)

    def _build_waypoints(self):
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link2)
        pos_obj = np.asarray(pos_obj, dtype=np.float32)
        orn_obj_euler = p.getEulerFromQuaternion(orn_obj)
        orn_eef = p.getEulerFromQuaternion(get_link_pose(self.psm2.body, self.psm2.EEF_LINK_INDEX)[1])

        roll = orn_obj_euler[0] - np.deg2rad(90)
        pitch = (
            orn_obj_euler[1]
            if abs(wrap_angle(orn_obj_euler[1] - orn_eef[1]))
            < abs(wrap_angle(orn_obj_euler[1] + np.pi - orn_eef[1]))
            else wrap_angle(orn_obj_euler[1] + np.pi)
        )
        yaw = (
            orn_obj_euler[2]
            if abs(wrap_angle(orn_obj_euler[2] - orn_eef[2]))
            < abs(wrap_angle(orn_obj_euler[2] + np.pi - orn_eef[2]))
            else wrap_angle(orn_obj_euler[2] + np.pi)
        )

        source_peg = self.block_source_peg_indices[0] if self.block_source_peg_indices else int(self._pegs[6])
        pos_peg = np.asarray(get_link_pose(self.obj_ids["fixed"][1], source_peg)[0], dtype=np.float32)
        pos_mid = np.asarray(
            [
                pos_obj[0],
                pos_obj[1] - pos_peg[1],
                pos_obj[2] + 0.043 * self.SCALING,
            ],
            dtype=np.float32,
        )

        rot_mat = np.asarray(p.getMatrixFromQuaternion(orn_obj)).reshape(3, 3)
        grasp_offset = np.array([-0.006, 0, 0], dtype=np.float32)
        grasp_pos = pos_obj + np.dot(rot_mat, grasp_offset)

        above_low = np.array(
            [
                grasp_pos[0],
                grasp_pos[1],
                pos_mid[2] - 0.02 * self.SCALING,
                roll,
                pitch,
                yaw,
                0.5,
            ],
            dtype=np.float32,
        )
        above_high = above_low.copy()
        above_high[2] = pos_mid[2] + 0.01 * self.SCALING

        approach = np.array(
            [
                grasp_pos[0],
                grasp_pos[1],
                grasp_pos[2] + (0.003 + 0.0102) * self.SCALING,
                roll,
                pitch,
                yaw,
                0.5,
            ],
            dtype=np.float32,
        )
        grasp = approach.copy()
        grasp[6] = -0.5
        lift = np.array(
            [grasp_pos[0], grasp_pos[1], pos_mid[2] + 0.01, roll, pitch, yaw, -0.5],
            dtype=np.float32,
        )
        return [above_high, above_low, approach, grasp, lift]

    def _reset_psm_to_waypoint(self, waypoint):
        orn = p.getQuaternionFromEuler(waypoint[3:6])
        joint_positions = self.psm2.inverse_kinematics(
            (waypoint[:3], orn),
            self.psm2.EEF_LINK_INDEX,
        )
        self.psm2.reset_joint(joint_positions)
        if waypoint[6] < 0.0:
            self.psm2.close_jaw()
        else:
            self.psm2.move_jaw(np.deg2rad(40))

    def _sample_waypoint_pos_noise(self):
        if self.waypoint_pos_noise_std <= 0.0:
            return np.zeros(3, dtype=np.float32)

        noise = np.random.normal(
            loc=0.0,
            scale=self.waypoint_pos_noise_std,
            size=3,
        ).astype(np.float32)
        if self.waypoint_pos_noise_clip > 0.0:
            noise = np.clip(noise, -self.waypoint_pos_noise_clip, self.waypoint_pos_noise_clip)
        if self.waypoint_noise_xy_only:
            noise[2] = 0.0
        return noise

    def _render_task_images(self):
        output = self.ecm.render_image(
            stereo=self.STEREO,
            scaling=self.SCALING,
            segmentation_with_link=True,
            FoV=60,
        )
        body_mask, link_mask = self._decode_segmentation_mask(output.mask1)
        peg_board_id = self.obj_ids["fixed"][1]
        # peg_mask = (body_mask == peg_board_id) & (link_mask >= 0)
        peg_mask = (body_mask == peg_board_id) & (link_mask == self.block_source_peg_indices[0])
        mask_no_arm = body_mask == self.target_id
        mask_target = (body_mask == self.psm2.body) | peg_mask | (body_mask == self.target_id)
        rgb2 = output.rgb2 if self.STEREO else output.rgb1
        return output.rgb1, rgb2, body_mask, mask_no_arm, mask_target, output.depth1

    @staticmethod
    def _decode_segmentation_mask(mask):
        mask = np.asarray(mask, dtype=np.int64)
        body_mask = mask.copy()
        link_mask = np.full(mask.shape, -1, dtype=np.int64)
        encoded_mask = mask >= (1 << LINK_ID_SHIFT)
        body_mask[encoded_mask] = mask[encoded_mask] & BODY_ID_MASK
        link_mask[encoded_mask] = (mask[encoded_mask] >> LINK_ID_SHIFT) - 1
        return body_mask, link_mask

    def project_to_pixel(self, centroid, img_shape):
        height, width = img_shape
        x_pixel = int((centroid[0] + 1) * (width / 2))
        y_pixel = int((centroid[1] + 1) * (height / 2))
        x_pixel = np.clip(x_pixel, 0, width - 1)
        y_pixel = np.clip(y_pixel, 0, height - 1)
        return np.array([x_pixel, y_pixel])

    def crop_around_pixel(self, rgb, center, crop_size=(64, 64)):
        x_c, y_c = center
        half_x = crop_size[1] // 2
        half_y = crop_size[0] // 2
        x1, x2 = max(0, x_c - half_x), min(rgb.shape[1], x_c + half_x)
        y1, y2 = max(0, y_c - half_y), min(rgb.shape[0], y_c + half_y)
        return rgb[y1:y2, x1:x2]

    def get_oracle_action(self, obs):
        rgb1, rgb2, mask, mask_no_arm, mask_target, depth = self._render_task_images()
        robot_state = self._get_robot_state()
        tip2_pos, _ = get_link_pose(self.psm2.body, self.psm2.TIP_LINK_INDEX)
        block_pos, _ = get_link_pose(self.obj_id, -1)
        tip2_pixel = self.project_to_pixel(self.ecm.get_centroid_proj(tip2_pos), rgb1.shape[:2])
        block_pixel = self.project_to_pixel(self.ecm.get_centroid_proj(block_pos), rgb1.shape[:2])
        tip2 = self.crop_around_pixel(rgb1, tip2_pixel)
        block = self.crop_around_pixel(rgb1, block_pixel)

        action = np.zeros(7, dtype=np.float32)
        action[6] = -0.5
        waypoint_idx = len(self._waypoints) - 1

        for waypoint_idx, waypoint in enumerate(self._waypoints):
            if self._waypoints_done[waypoint_idx]:
                continue

            delta_pos = (waypoint[:3] - obs["observation"][:3]) / (0.01 * self.SCALING)
            if np.abs(delta_pos).max() > 1:
                delta_pos = delta_pos / np.abs(delta_pos).max()
            scale_factor = 0.3
            delta_pos *= scale_factor
            delta_rpy = np.array(
                [
                    wrap_angle(waypoint[3] - obs["observation"][3]),
                    wrap_angle(waypoint[4] - obs["observation"][4]),
                    wrap_angle(waypoint[5] - obs["observation"][5]),
                ],
                dtype=np.float32,
            )
            delta_rpy = np.clip(delta_rpy, -1.0, 1.0)
            action = np.array(
                [
                    delta_pos[0],
                    delta_pos[1],
                    delta_pos[2],
                    delta_rpy[0],
                    delta_rpy[1],
                    delta_rpy[2],
                    waypoint[6],
                ],
                dtype=np.float32,
            )

            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3:
                self._waypoints_done[waypoint_idx] = True
            break

        return (
            action,
            rgb1,
            rgb2,
            mask,
            mask_no_arm,
            mask_target,
            depth,
            robot_state,
            waypoint_idx,
            tip2.copy(),
            tip2,
            block,
            [tip2_pixel.copy(), tip2_pixel, block_pixel],
        )


if __name__ == "__main__":
    env = PegBlockPickPsm2(render_mode="human", action_mode="rpy")
    obs = env.reset()
    for _ in range(200):
        oracle = env.get_oracle_action(obs)
        action = oracle[0] if isinstance(oracle, tuple) else oracle
        obs, _, _, info = env.step(action)
        if bool(np.asarray(info["is_success"])):
            break
    env.close()
    time.sleep(2)