from gym.envs.registration import register


# PSM Env
register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieve-v0',
    entry_point='surrol.tasks.gauze_retrieve:GauzeRetrieve',
    max_episode_steps=50,
)

register(
    id='NeedlePick-v0',
    entry_point='surrol.tasks.needle_pick:NeedlePick',
    max_episode_steps=100,
)

register(
    id='NeedlePick-v1',
    entry_point='surrol.tasks.needle_pick_kj_edition:NeedlePick',
    max_episode_steps=100,
)


register(
    id='PegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransfer-v1',
    entry_point='surrol.tasks.peg_transfer_rpy:PegTransfer',
    max_episode_steps=200,
)
register(
    id='PegTransfer-v2',
    entry_point='surrol.tasks.peg_transfer_rpy_new:PegTransfer',
    max_episode_steps=200,
)
register(
    id='PegTransfer-v3',
    entry_point='surrol.tasks.peg_transfer_rpy_new_foveal:PegTransfer',
    max_episode_steps=200,
)
register(
    id='PegTransfer-v4',
    entry_point='surrol.tasks.peg_transfer_rpy_new_foveal_edit:PegTransfer',
    max_episode_steps=200,
)
register(
    id='PegTransfer-v5',
    entry_point='surrol.tasks.peg_transfer_rpy_new_foveal_edit_all_fixed:PegTransfer',
    max_episode_steps=200,
)
# Bimanual PSM Env
register(
    id='NeedleRegrasp-v0',
    entry_point='surrol.tasks.needle_regrasp_bimanual:NeedleRegrasp',
    max_episode_steps=50,
)

register(
    id='BiPegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer_bimanual_0120:BiPegTransfer',
    max_episode_steps=120,
)

register(
    id='BiPegTransfer-v1',
    entry_point='surrol.tasks.peg_transfer_bimanual:BiPegTransfer',
    max_episode_steps=120,
)

register(
    id='BiPegTransfer-v2',
    entry_point='surrol.tasks.peg_transfer_bimanual_new:BiPegTransfer',
    max_episode_steps=200,
)

register(
    id='BiPegTransfer-v3',
    entry_point='surrol.tasks.peg_transfer_bimanual_new_add_foveal:BiPegTransfer',
    max_episode_steps=200,
)
register(
    id='BiPegTransfer-v4',
    entry_point='surrol.tasks.peg_transfer_bimanual_new_add_foveal_all_fixed:BiPegTransfer',
    max_episode_steps=200,
)
# ECM Env
register(
    id='ECMReach-v0',
    entry_point='surrol.tasks.ecm_reach:ECMReach',
    max_episode_steps=50,
)

register(
    id='MisOrient-v0',
    entry_point='surrol.tasks.ecm_misorient:MisOrient',
    max_episode_steps=50,
)

register(
    id='StaticTrack-v0',
    entry_point='surrol.tasks.ecm_static_track:StaticTrack',
    max_episode_steps=50,
)

register(
    id='ActiveTrack-v0',
    entry_point='surrol.tasks.ecm_active_track:ActiveTrack',
    max_episode_steps=500,
)
