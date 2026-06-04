# DP3 on SurRoL BiPegTransfer

This folder starts the DP3-style single-view point cloud pipeline for the current
SurRoL `BiPegTransfer` demonstrations.

For the full end-to-end migration plan from collected demos to DP3 training and
SurRoL online evaluation, see `WORKFLOW_SURROL_BIPEG_DP3.md`.

The existing pickle demos already contain `rgb1`, `depths`, masks, `qpos`, and
14-D bimanual actions. The converter turns each `rgb1 + depth + mask` frame into
a fixed-size camera-frame point cloud.

## One-Command Pipeline

The embedded DP3 checkout now has a single pipeline script that can reuse or
rebuild the point-cloud NPZ, inspect DP3 batches, train a policy, evaluate it in
`BiPegTransfer-v4`, and save rollout videos:

```bash
cd ~/Surrol

MODEL_SIZE=tiny TRAIN_EPOCHS=50 BATCH_SIZE=8 EVAL_EPISODES=20 MAX_STEPS=200 \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/run_surrol_bipeg_transfer_pipeline.sh
```

Important knobs:

- `DATA_NPZ`: point-cloud dataset, default `collected_data/bipeg_transfer_dp3_pointcloud.npz`.
- `RUN_CONVERT=True`: force pickle-to-NPZ conversion before training.
- `MODEL_SIZE=tiny|small|base`: select the DP3 model size.
- `DEVICE=cpu` or `DEVICE=cuda:0`: override auto device detection.
- `RUN_TAG=train50`: output run name suffix.
- `RUN_TRAIN=False` or `RUN_EVAL=False`: skip one half of the pipeline.

Fast smoke test:

```bash
cd ~/Surrol

RUN_TAG=smoke MODEL_SIZE=tiny TRAIN_EPOCHS=1 MAX_TRAIN_STEPS=2 \
SAMPLE_EVERY=-1 BATCH_SIZE=2 EVAL_EPISODES=1 MAX_STEPS=5 DEVICE=cpu \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/run_surrol_bipeg_transfer_pipeline.sh
```

The outputs are written under:

```text
Methods/dp3-surrol/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/
  surrol_bipeg_transfer-dp3-<MODEL_SIZE>-<RUN_TAG>_seed0/
    checkpoints/latest.ckpt
    eval_outputs/results.json
    eval_outputs/episodes.csv
    eval_outputs/videos/episode_000.mp4
```

## Convert Pickle Demos

From the repository root, or from this `Methods/dp3-surrol` folder:

```bash
python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py \
  --input-dir collected_data/bipeg_transfer \
  --output collected_data/bipeg_transfer_dp3_pointcloud.npz \
  --num-points 1024 \
  --mask-mode target \
  --action-mode absolute
```

Useful variants:

```bash
# Keep per-point RGB as a separate point_color array.
python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py --include-rgb

# Use the full rendered depth image instead of target masks.
python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py --mask-mode all

# Emit a DP3-style zarr tree if zarr is installed.
python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py \
  --output collected_data/bipeg_transfer_dp3_pointcloud.zarr \
  --output-format zarr

# Emit one npz file per source episode.
python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py \
  --output collected_data/bipeg_transfer_dp3_episodes \
  --output-format episode_npz
```

## Output Schema

The `.npz` output contains:

- `point_cloud`: `(T, N, 3)` float32 camera-frame XYZ points.
- `point_color`: `(T, N, 3)` float32 RGB values in `[0, 1]`, only with `--include-rgb`.
- `action`: `(T, 14)` float32 bimanual action targets.
- `agent_pos`: `(T, 14)` float32 bimanual robot state from `qpos`.
- `waypoint`: `(T,)` int32 source waypoint id.
- `episode_id`: `(T,)` int32 source pickle episode id.
- `episode_ends`: `(E,)` int64 cumulative episode ends for replay-buffer style loaders.
- `camera_intrinsic`: `(3, 3)` float32 pinhole intrinsic matrix.
- `metadata`: JSON string with conversion settings.

The default `npz` mode stores all frames in one replay-buffer style file. Use
`episode_npz` mode only when your downstream loader expects one file per
episode.

## Visualize Point Clouds

```bash
python3 Methods/dp3-surrol/visualize_pointcloud_npz.py \
  --input collected_data/bipeg_transfer_dp3_pointcloud.npz \
  --output collected_data/bipeg_transfer_dp3_pointcloud_preview.png \
  --episode 0 \
  --num-frames 5
```

This saves a static preview with 3D point clouds and XY projections sampled from
the selected episode. The XY projection inverts the displayed Y axis by default
to match image coordinates, where pixel Y increases downward. Use
`--no-invert-y` to inspect raw camera-frame Y values. Use `--display-y-up` for
a human-upright display without modifying the `.npz` data.

## Render Point Cloud Video

```bash
python3 Methods/dp3-surrol/video_pointcloud_npz.py \
  --input collected_data/bipeg_transfer_dp3_pointcloud.npz \
  --output collected_data/bipeg_transfer_dp3_pointcloud_episode0.mp4 \
  --episode 0 \
  --fps 12
```

This renders one episode as a side-by-side video: 3D camera-frame point cloud on
the left and XY projection on the right. Add `--spin` to rotate the 3D view, or
use `--max-frames`, `--start-frame`, `--end-frame`, and `--stride` to shorten
the output. Add `--display-y-up` if you want the video to look upright in a
typical plotting/world-style coordinate view.

## Inspect DP3 Batches

```bash
python3 Methods/dp3-surrol/inspect_dataset_batch.py \
  --data collected_data/bipeg_transfer_dp3_pointcloud.npz \
  --split both \
  --batch-size 4 \
  --obs-horizon 2 \
  --action-horizon 16
```

The dataset adapter in `surrol_bipeg_dataset.py` maps the replay-buffer NPZ into
causal DP3-style windows:

- `obs.point_cloud`: `(B, obs_horizon, 1024, 3)`
- `obs.agent_pos`: `(B, obs_horizon, 14)`
- `action`: `(B, action_horizon, 14)`

The observation window uses the current and previous frames with episode-boundary
padding. The action window starts at the current frame and pads with the final
frame near the episode end. The train/val split is episode-level, not random
frame-level, to avoid leakage.

`configs/bipegtransfer_dp3_pointcloud.yaml` is a config draft you can port into
the actual DP3 training repo. If that repo uses Hydra class targets, add this
folder to `PYTHONPATH` and point the dataset target to
`surrol_bipeg_dataset.SurrolBipegSequenceDataset`.

## Train With Local DP3 Repo

The local DP3 checkout at `Methods/dp3-surrol/3D-Diffusion-Policy` now has a
SurRoL task config:

```bash
bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer_debug.sh 0 0
```

This launches `dp3.yaml` with `task=surrol_bipeg_transfer`, disabled WandB mode,
small debug batch size, and the NPZ dataset at
`collected_data/bipeg_transfer_dp3_pointcloud.npz`. The script defaults to a
tiny DP3 model to make the first SurRoL experiments practical:

- `MODEL_SIZE=tiny`: overrides `down_dims=[128,256,512]`,
  `encoder_output_dim=32`, and `diffusion_step_embed_dim=64`.
- `MODEL_SIZE=small`: overrides `down_dims=[256,512,1024]` and keeps the
  original encoder/timestep embedding sizes.
- `MODEL_SIZE=base`: uses the original `dp3.yaml` model, about 255M parameters.

Useful examples:

```bash
# Fast smoke/debug run, 10 train steps per epoch because DEBUG=True.
MODEL_SIZE=tiny WANDB_MODE=disabled BATCH_SIZE=8 \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer_debug.sh 0 0 debug

# Add VERBOSE_TIMING=True only when you want per-step timing prints.
MODEL_SIZE=tiny VERBOSE_TIMING=True WANDB_MODE=disabled BATCH_SIZE=8 \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer_debug.sh 0 0 timing

# Short non-debug training run with full epochs and latest.ckpt saving.
MODEL_SIZE=tiny NUM_EPOCHS=50 WANDB_MODE=disabled BATCH_SIZE=8 \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer.sh 0 0 train50

# Evaluate the saved policy in SurRoL BiPegTransfer-v4 and save rollout videos.
MODEL_SIZE=tiny EVAL_EPISODES=20 OUTPUT_VIDEO=True WANDB_MODE=disabled \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh 0 0 train50
```

The scripts auto-detect `cuda:0` when CUDA is available and otherwise fall back
to CPU. Override with `DEVICE=cuda:0` or `DEVICE=cpu`. The pipeline script also
tries to activate conda env `dp3` automatically; set `AUTO_CONDA=False` if you
already activated the desired environment. Online SurRoL eval defaults to
PyBullet TinyRenderer (`SURROL_RENDERER=tiny`, `SURROL_USE_EGL=0`) so headless
machines do not have to load the EGL plugin.

The embedded DP3 config returns horizon-length windows:

- `obs.point_cloud`: `(B, horizon, 1024, 3)`
- `obs.agent_pos`: `(B, horizon, 14)`
- `action`: `(B, horizon, 14)`

DP3 then uses only the first `n_obs_steps=2` observations as conditioning and
trains the diffusion model over `horizon=16` actions.

Minimal offline-training environment:

```bash
conda create -n dp3 python=3.8 -y
conda activate dp3

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd Methods/dp3-surrol/3D-Diffusion-Policy/3D-Diffusion-Policy
pip install -e .
cd ..

pip install numpy==1.23.5 zarr==2.12.0 wandb ipdb gpustat \
  omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 \
  diffusers==0.11.1 moviepy imageio av matplotlib \
  termcolor tqdm huggingface_hub==0.25.2 six
```

This minimal environment is for the SurRoL DP3 path in this folder. It does not
install Mujoco, MetaWorld, or DexArt.

## Camera Assumptions

The point cloud is currently in the ECM camera frame. This is intentional: the
pickle files store depth images but do not store the ECM camera pose/view matrix
for each frame.

For this dataset, the converter defaults to the rendering parameters in
`Simulator/surrol/robots/ecm.py`:

- image size: `256x256`
- field of view: `60` degrees
- focal length: about `221.70`
- principal point: `(128, 128)`

If you want world-frame point clouds, update the data collection loop to save the
ECM camera pose or `view_matrix` for each timestep, then transform these camera
points into the simulator/world frame.

## DP3 Integration Plan

1. Convert the existing 50 pickle demos with this script.
2. Add a DP3 dataset loader that maps:
   - `obs.point_cloud <- point_cloud`
   - `obs.agent_pos <- agent_pos`
   - `action <- action`
3. Train a DP3 policy with `shape_meta` matching `point_cloud: [1024, 3]`,
   `agent_pos: [14]`, and `action: [14]`.
4. Add a SurRoL eval wrapper that renders `rgb1/depth/mask` at each timestep,
   samples the same point cloud format, and sends predicted 14-D actions to
   `BiPegTransfer-v4`.

Recommended first baseline:

- `--mask-mode target`
- `--action-mode absolute`
- `--num-points 1024`
- camera-frame point clouds

Use `--action-mode dq` only for an ablation that also has a matching
dq-to-control execution path. For direct DP3 rollout in SurRoL, `absolute` is
the safer first target because it matches the oracle action sent to
`env.step(action)`.

After that baseline works, compare against `--mask-mode all` and larger point
sets such as `2048`.
