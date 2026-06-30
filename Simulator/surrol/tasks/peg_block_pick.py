import os

import numpy as np
import pybullet as p

from Simulator.surrol.tasks.peg_transfer_bimanual_new_add_foveal_all_fixed import (
    BiPegTransfer,
    _parse_bool,
)
from Simulator.surrol.utils.pybullet_utils import get_link_pose


def _parse_float(value, default):
    if value is None:
        return float(default)
    return float(value)


class PegBlockPick(BiPegTransfer):
    """PSM2-only block picking subtask from BiPegTransfer-v4.

    The task keeps the same 14D bimanual action/state convention as the full
    BiPegTransfer task. On reset, PSM2 is placed at the original BiPegTransfer
    waypoint[2] plus optional position noise. The oracle then executes the
    clean grasp and lift segment from that perturbed reset pose. PSM1 stays at
    the waiting pose encoded in the inherited waypoints.
    """

    DEFAULT_WAYPOINT2_POS_NOISE_STD = 0.01 * BiPegTransfer.SCALING
    DEFAULT_WAYPOINT2_POS_NOISE_CLIP = 0.03 * BiPegTransfer.SCALING
    DEFAULT_WAYPOINT2_NOISE_XY_ONLY = True
    LIFT_SUCCESS_HEIGHT = 0.0175 * BiPegTransfer.SCALING

    def __init__(
        self,
        render_mode=None,
        cid=-1,
        action_mode="yaw",
        randomize_block_initial_position=None,
        block_initial_peg_indices=None,
        block_initial_xy_jitter=None,
        waypoint2_pos_noise_std=None,
        waypoint2_pos_noise_clip=None,
        waypoint2_noise_xy_only=None,
    ):
        if waypoint2_pos_noise_std is None:
            waypoint2_pos_noise_std = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_WAYPOINT2_POS_NOISE_STD",
                self.DEFAULT_WAYPOINT2_POS_NOISE_STD,
            )
        if waypoint2_pos_noise_clip is None:
            waypoint2_pos_noise_clip = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_WAYPOINT2_POS_NOISE_CLIP",
                self.DEFAULT_WAYPOINT2_POS_NOISE_CLIP,
            )
        if waypoint2_noise_xy_only is None:
            waypoint2_noise_xy_only = os.environ.get(
                "SURROL_PEG_BLOCK_PICK_WAYPOINT2_NOISE_XY_ONLY",
                self.DEFAULT_WAYPOINT2_NOISE_XY_ONLY,
            )

        self.waypoint2_pos_noise_std = _parse_float(
            waypoint2_pos_noise_std,
            self.DEFAULT_WAYPOINT2_POS_NOISE_STD,
        )
        self.waypoint2_pos_noise_clip = _parse_float(
            waypoint2_pos_noise_clip,
            self.DEFAULT_WAYPOINT2_POS_NOISE_CLIP,
        )
        self.waypoint2_noise_xy_only = _parse_bool(
            waypoint2_noise_xy_only,
            default=self.DEFAULT_WAYPOINT2_NOISE_XY_ONLY,
        )
        self.initial_block_pos = None
        self.last_waypoint2_pos_noise = np.zeros(3, dtype=np.float32)
        self.reset_waypoint = None

        super().__init__(
            render_mode=render_mode,
            cid=cid,
            action_mode=action_mode,
            randomize_block_initial_position=randomize_block_initial_position,
            block_initial_peg_indices=block_initial_peg_indices,
            block_initial_xy_jitter=block_initial_xy_jitter,
        )
        print("begin peg-block-pick")

    def _env_setup(self):
        super()._env_setup()
        block_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
        self.initial_block_pos = np.asarray(block_pos, dtype=np.float32)

    def _sample_goal_callback(self):
        """Reset PSM2 at noisy waypoint[2], then use clean grasp-lift targets."""
        super()._sample_goal_callback()

        full_waypoints = [waypoint.copy() for waypoint in self._waypoints]
        self.last_waypoint2_pos_noise = self._sample_waypoint2_pos_noise()
        full_waypoints[1][13] = -0.5
        self.reset_waypoint = full_waypoints[1].copy()
        self.reset_waypoint[7:10] += self.last_waypoint2_pos_noise
        self.reset_waypoint[9] += 0.01 * BiPegTransfer.SCALING

        self._reset_arms_to_waypoint(self.reset_waypoint)
        self._waypoints = [
            # full_waypoints[0],  # PSM2 init
            full_waypoints[1],  # PSM2 ab
            full_waypoints[2],  # PSM2 approach
            full_waypoints[3],  # PSM2 clean grasp pose, jaw closed
            full_waypoints[4],  # PSM2 lift after grasp
        ]
        self._waypoints_done = [False] * len(self._waypoints)

    def _reset_arms_to_waypoint(self, waypoint):
        self._reset_psm_to_waypoint(self.psm1, waypoint[0:3], waypoint[3:6], waypoint[6])
        self._reset_psm_to_waypoint(self.psm2, waypoint[7:10], waypoint[10:13], waypoint[13])

    def _reset_psm_to_waypoint(self, psm, pos, euler, jaw):
        orn = p.getQuaternionFromEuler(euler)
        joint_positions = psm.inverse_kinematics(
            (pos, orn),
            psm.EEF_LINK_INDEX,
        )
        psm.reset_joint(joint_positions)
        if jaw < 0.0:
            psm.close_jaw()
        else:
            psm.move_jaw(np.deg2rad(40))

    def _sample_waypoint2_pos_noise(self):
        if self.waypoint2_pos_noise_std <= 0.0:
            return np.zeros(3, dtype=np.float32)

        noise = np.random.normal(
            loc=0.0,
            scale=self.waypoint2_pos_noise_std,
            size=3,
        ).astype(np.float32)

        if self.waypoint2_pos_noise_clip > 0.0:
            noise = np.clip(
                noise,
                -self.waypoint2_pos_noise_clip,
                self.waypoint2_pos_noise_clip,
            )

        if self.waypoint2_noise_xy_only:
            noise[2] = 0.0

        return noise

    def _is_success(self, achieved_goal, desired_goal):
        """Success means the block has been lifted by PSM2."""
        if self.initial_block_pos is None:
            return np.zeros_like(achieved_goal[..., -1], dtype=np.float32)

        block_pos, _ = get_link_pose(self.obj_id, -1)
        lifted = block_pos[2] > self.initial_block_pos[2] + self.LIFT_SUCCESS_HEIGHT
        grasped_by_psm2 = self._activated == 1
        return np.asarray(lifted and grasped_by_psm2, dtype=np.float32)
