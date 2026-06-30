
### 生成数据

peg_transfer_bimanual_new_add_foveal_all_fixed.py 里控制块初始化时所在的 peg 位置，以及块的 xy 坐标扰动值
data_generation_foveal.py 里控制成功收集的数据组数

python3 Simulator/surrol/data/data_generation_foveal.py \
  --env BiPegTransfer-v4 \
  --randomize-block-initial-position \
  --block-initial-xy-jitter 0.01

python3 Simulator/surrol/data/data_generation_foveal.py \
  --env PegBlockPickPsm2-v0


### pkl 转点云

pkl转全局点云，--mask-mode all 包括整张深度图里可见的臂、块、peg、板子和背景
python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py \
  --input-dir collected_data/bipeg_transfer \
  --output collected_data/bipeg_transfer_dp3_global_target_fixedpeg_xyjitter_pointcloud.npz \
  --num-points 1024 \
  --mask-mode all \
  --point-noise-std 0 \
  --point-noise-clip 0 \
  --depth-noise-std 0.003 \
  --depth-noise-clip 0.01 \
  --action-mode absolute

把 --mask-mode 改成 target 后，点云只包括臂，块，peg  
python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py \
  --input-dir collected_data/peg_block_pick \
  --output collected_data/peg_block_pick_dp3_waypoint0noise.npz \
  --num-points 1024 \
  --mask-mode target \
  --action-mode absolute

### 点云可视化
python3 Methods/dp3-surrol/video_pointcloud_npz.py \
  --input collected_data/bipeg_transfer_dp3_global_target_noise003.npz \
  --output collected_data/bipeg_transfer_dp3_global_target_noise003.mp4 \
  --episode 0 \
  --fps 12

python3 Methods/dp3-surrol/video_pointcloud_npz.py \
  --input collected_data/peg_block_pick_dp3_waypoint1xynoise.npz \
  --output collected_data/peg_block_pick_dp3_waypoint1xynoise.mp4 \
  --episode 0 \
  --fps 12


完整流水线，maskmode同样可以改
RUN_CONVERT=True \
MODEL_SIZE=small \
NUM_POINTS=1024 \
MASK_MODE=all \
POINT_NOISE_STD=0 \
POINT_NOISE_CLIP=0 \
DEPTH_NOISE_STD=0.003 \
DEPTH_NOISE_CLIP=0.01 \
RANDOMIZE_BLOCK_INITIAL_POSITION=False \
BLOCK_INITIAL_PEGS=6 \
BLOCK_INITIAL_XY_JITTER=0.002 \
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/run_surrol_bipeg_transfer_pipeline.sh

RUN_CONVERT=True \
MODEL_SIZE=tiny \
NUM_POINTS=1024 \
MASK_MODE=target \
POINT_NOISE_STD=0 \
POINT_NOISE_CLIP=0 \
DEPTH_NOISE_STD=0.003 \
DEPTH_NOISE_CLIP=0.01 \
RANDOMIZE_BLOCK_INITIAL_POSITION=False \
BLOCK_INITIAL_PEGS=6 \
BLOCK_INITIAL_XY_JITTER=0.002 \
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/run_surrol_bipeg_transfer_pipeline.sh

关掉随机初始位置
RANDOMIZE_BLOCK_INITIAL_POSITION=False \
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh 0 1000 train50

关掉在线噪声
POINT_NOISE_STD=0 \
POINT_NOISE_CLIP=0 \
DEPTH_NOISE_STD=0 \
DEPTH_NOISE_CLIP=0 \
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh 0 1000 train50

固定peg6
RANDOMIZE_BLOCK_INITIAL_POSITION=False \
BLOCK_INITIAL_PEGS=6 \
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh 0 1000 train50



现在 target mask 包含双臂、peg board 的所有 link 和 block；但 DP3 实际训练/评估用的是全局点云 mask_mode=all。block 初始 peg 固定在 6，只对初始 x/y 和 yaw 做轻微扰动，同时点云还会加高斯噪声

python3 Simulator/surrol/data/data_generation_foveal.py \
  --env BiPegTransfer-v4 \
  --no-randomize-block-initial-position \
  --block-initial-pegs 6 \
  --block-initial-xy-jitter 0.002


python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py \
  --input-dir collected_data/bipeg_transfer \
  --output collected_data/bipeg_transfer_dp3_global_target_fixedpeg_xyjitter_pointcloud.npz \
  --num-points 1024 \
  --mask-mode target \
  --point-noise-std 0 \
  --point-noise-clip 0 \
  --depth-noise-std 0.003 \
  --depth-noise-clip 0.01 \
  --action-mode absolute


RUN_CONVERT=True \
MODEL_SIZE=tiny \
NUM_POINTS=1024 \
MASK_MODE=target \
RANDOMIZE_BLOCK_INITIAL_POSITION=False \
BLOCK_INITIAL_PEGS=6 \
BLOCK_INITIAL_XY_JITTER=0.002 \
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/run_surrol_bipeg_transfer_pipeline.sh



python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py \
  --input-dir collected_data/bipeg_transfer_blockinpeg6_targetwithpeg \
  --output collected_data/bipeg_transfer_dp3_pointcloud_1024_depthnoise_0mm.npz \
  --num-points 1024 \
  --mask-mode target \
  --action-mode absolute \
  --depth-noise-std 0.001 \
  --depth-noise-clip 0.003 


### 深度图噪声

cd /home/escthought/Surrol/Methods/dp3-surrol/3D-Diffusion-Policy/3D-Diffusion-Policy

SURROL_PEG_BLOCK_PICK_WAYPOINT2_POS_NOISE_STD=0.15 \
SURROL_PEG_BLOCK_PICK_WAYPOINT2_POS_NOISE_CLIP=0.5 \
CUDA_VISIBLE_DEVICES=0 python3 eval.py --config-name=dp3.yaml \
  task=surrol_bipeg_transfer \
  hydra.run.dir=data/outputs/pegblockpick-dp3-tiny-clean50_seed9982 \
  training.seed=9982 \
  training.device=cuda:0 \
  logging.mode=disabled \
  "task.shape_meta.obs.point_cloud.shape=[1024,3]" \
  task.env_runner.env_id=PegBlockPick-v0 \
  task.env_runner.eval_episodes=20 \
  task.env_runner.max_steps=80 \
  task.env_runner.num_points=1024 \
  task.env_runner.mask_mode=target \
  task.env_runner.point_noise_std=0 \
  task.env_runner.point_noise_clip=0 \
  task.env_runner.depth_noise_std=0 \
  task.env_runner.depth_noise_clip=0 \
  task.env_runner.randomize_block_initial_position=false \
  "task.env_runner.block_initial_peg_indices=[6]" \
  task.env_runner.block_initial_xy_jitter=0 \
  task.env_runner.output_video=true \
  'policy.down_dims=[128,256,512]' \
  policy.encoder_output_dim=32 \
  policy.diffusion_step_embed_dim=64




### waypoint1 噪声

cd /home/escthought/Surrol

SURROL_PEG_BLOCK_PICK_WAYPOINT2_POS_NOISE_STD=0 \
SURROL_PEG_BLOCK_PICK_WAYPOINT2_POS_NOISE_CLIP=0 \
DATA_NPZ=collected_data/peg_block_pick_dp3_waypoint1xynoise2.npz \
RUN_CONVERT=True \
RUN_INSPECT=True \
RUN_TRAIN=True \
RUN_EVAL=True \
ENV_ID=PegBlockPick-v0 \
SEED=9982 \
RUN_TAG=waypoint1xynoise2_50 \
MODEL_SIZE=tiny \
NUM_POINTS=1024 \
MASK_MODE=target \
POINT_NOISE_STD=0 \
POINT_NOISE_CLIP=0 \
DEPTH_NOISE_STD=0 \
DEPTH_NOISE_CLIP=0 \
RANDOMIZE_BLOCK_INITIAL_POSITION=False \
BLOCK_INITIAL_PEGS=6 \
BLOCK_INITIAL_XY_JITTER=0 \
WANDB_MODE=disabled \
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/run_surrol_bipeg_transfer_pipeline.sh


TASK_CONFIG=surrol_peg_block_pick_psm2 RUN_PREFIX=peg_block_pick_psm2_v2 data_npz=peg_block_pick_psm2_v2_pointcloud.npz MODEL_SIZE=tiny NUM_EPOCHS=50 bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer.sh 0 0 train50

TASK_CONFIG=surrol_peg_block_pick_psm2 RUN_PREFIX=peg_block_pick_psm2_v2 MODEL_SIZE=tiny bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh   0 0 train50   task.env_runner.save_action_trace=true   task.env_runner.trace_success_episodes=true


TASK_CONFIG=surrol_peg_block_pick_psm2 RUN_PREFIX=peg_block_pick_psm2_cropped_v2 data_npz=peg_block_pick_psm2_cropped_v2_pointcloud.npz MODEL_SIZE=tiny NUM_EPOCHS=50 bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer.sh 0 9982 train50

TASK_CONFIG=surrol_peg_block_pick_psm2 RUN_PREFIX=peg_block_pick_psm2_cropped_v2 MODEL_SIZE=tiny bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh   0 9982 train50   task.env_runner.save_action_trace=true   task.env_runner.trace_success_episodes=true



TASK_CONFIG=surrol_peg_block_pick_psm2 RUN_PREFIX=peg_block_pick_psm2_cropped_v2 data_npz=peg_block_pick_psm2_cropped_v2_pointcloud.npz MODEL_SIZE=tiny NUM_EPOCHS=50 bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer.sh 0 44353 train50

TASK_CONFIG=surrol_peg_block_pick_psm2 RUN_PREFIX=peg_block_pick_psm2_cropped_v2 MODEL_SIZE=tiny bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh   0 44353 train50   task.env_runner.save_action_trace=true   task.env_runner.trace_success_episodes=true


TASK_CONFIG=surrol_peg_block_pick_psm2 \
RUN_PREFIX=peg_block_pick_psm2_cropped_v2 \
DATA_NPZ=collected_data/peg_block_pick_psm2_cropped_v2_pointcloud.npz \
MODEL_SIZE=small \
NUM_EPOCHS=50 \
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer.sh 0 44353 train50


  TASK_CONFIG=surrol_peg_block_pick_psm2 \
  RUN_PREFIX=peg_block_pick_psm2_cropped_v2 \
  MODEL_SIZE=small \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh 0 44353 train50
