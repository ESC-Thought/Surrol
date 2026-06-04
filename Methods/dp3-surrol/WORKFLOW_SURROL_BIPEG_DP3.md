# SurRoL BiPegTransfer + 3D Diffusion Policy Workflow

目标是把 `3D-Diffusion-Policy` 移植到 SurRoL 的 `BiPegTransfer-v4`
任务上，形成一条单一闭环：

```text
collected_data pickle demos
  -> single-view point-cloud NPZ
  -> DP3 offline training
  -> checkpointed policy
  -> SurRoL BiPegTransfer online rollout
  -> success rate / reward / video / JSON results
```

## 0. 当前已有内容

已有数据和工具：

- 原始演示数据：`collected_data/bipeg_transfer/image_action_qpos_*.pkl`
- 点云转换器：`Methods/dp3-surrol/convert_bipegtransfer_pickles.py`
- 转换后数据：`collected_data/bipeg_transfer_dp3_pointcloud.npz`
- 外层 batch 检查器：`Methods/dp3-surrol/inspect_dataset_batch.py`
- 外层 dataset 草稿：`Methods/dp3-surrol/surrol_bipeg_dataset.py`
- DP3 原仓库：`Methods/dp3-surrol/3D-Diffusion-Policy/3D-Diffusion-Policy`

当前实现状态：

已落地：

- DP3 内部 dataset：`diffusion_policy_3d/dataset/surrol_bipeg_dataset.py`
- DP3 task config：`diffusion_policy_3d/config/task/surrol_bipeg_transfer.yaml`
- SurRoL train 脚本：`scripts/train_surrol_bipeg_transfer.sh`
- SurRoL debug train 脚本：`scripts/train_surrol_bipeg_transfer_debug.sh`
- SurRoL online runner：`diffusion_policy_3d/env_runner/surrol_bipeg_runner.py`
- SurRoL eval 脚本：`scripts/eval_surrol_bipeg_transfer.sh`
- 端到端 pipeline：`scripts/run_surrol_bipeg_transfer_pipeline.sh`
- Headless PyBullet 修复：SurRoL DIRECT 默认不强制 EGL，在线评估默认使用
  TinyRenderer；需要 EGL 时可显式设置 `SURROL_USE_EGL=1`。

一键入口：

```bash
cd ~/Surrol

MODEL_SIZE=tiny TRAIN_EPOCHS=50 BATCH_SIZE=8 EVAL_EPISODES=20 MAX_STEPS=200 \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/run_surrol_bipeg_transfer_pipeline.sh
```

快速链路验证：

```bash
cd ~/Surrol

RUN_TAG=smoke MODEL_SIZE=tiny TRAIN_EPOCHS=1 MAX_TRAIN_STEPS=2 \
SAMPLE_EVERY=-1 BATCH_SIZE=2 EVAL_EPISODES=1 MAX_STEPS=5 DEVICE=cpu \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/run_surrol_bipeg_transfer_pipeline.sh
```

输出位置：

```text
Methods/dp3-surrol/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/
  surrol_bipeg_transfer-dp3-<MODEL_SIZE>-<RUN_TAG>_seed0/
    checkpoints/latest.ckpt
    eval_outputs/results.json
    eval_outputs/episodes.csv
    eval_outputs/videos/episode_000.mp4
```

## 1. 数据规范

先把 pickle demo 固定转换成 DP3 可读的 replay-buffer NPZ：

```bash
python3 Methods/dp3-surrol/convert_bipegtransfer_pickles.py \
  --input-dir collected_data/bipeg_transfer \
  --output collected_data/bipeg_transfer_dp3_pointcloud.npz \
  --num-points 1024 \
  --mask-mode target \
  --action-mode absolute
```

输出约定：

- `point_cloud`: `(T, 1024, 3)`，ECM camera-frame XYZ。
- `agent_pos`: `(T, 14)`，双臂 robot state。
- `action`: `(T, 14)`，oracle 传给 `env.step(action)` 的 14-D 控制命令。
- `episode_ends`: `(E,)`，每条 episode 的 cumulative end index。
- `metadata`: conversion 参数，必须记录 `mask_mode/action_mode/num_points`。

第一版 baseline 固定使用：

```text
mask_mode = target
action_mode = absolute
num_points = 1024
point_cloud_frame = camera
```

注意：这里的 `absolute` 不是“绝对关节角”，而是使用 pickle 中的
`actions` 字段；它和 data collection 里执行的 `env.step(action)` 对齐。
不要先用 `dq_actions`，除非同时实现 dq 到 SurRoL 控制命令的执行逻辑。

## 2. DP3 Dataset 移植

把外层 `Methods/dp3-surrol/surrol_bipeg_dataset.py` 的核心逻辑移到 DP3 包内：

```text
Methods/dp3-surrol/3D-Diffusion-Policy/3D-Diffusion-Policy/
  diffusion_policy_3d/dataset/surrol_bipeg_dataset.py
```

DP3 内部 dataset 应该：

- 继承 `diffusion_policy_3d.dataset.base_dataset.BaseDataset`。
- 读取 `collected_data/bipeg_transfer_dp3_pointcloud.npz`。
- 按 episode-level split 划分 train/val，避免同一条 demo 泄漏到 train 和 val。
- 返回 DP3 horizon 窗口：
  - `obs.point_cloud`: `(horizon, 1024, 3)`
  - `obs.agent_pos`: `(horizon, 14)`
  - `action`: `(horizon, 14)`
- 实现 `get_validation_dataset()`。
- 实现 `get_normalizer()`，至少覆盖 `point_cloud/agent_pos/action`。

DP3 的 policy 会用 `n_obs_steps=2` 作为条件；dataset 返回完整
`horizon=16` 是为了和原 DP3 训练接口一致。

## 3. DP3 Task Config

新增任务配置：

```text
diffusion_policy_3d/config/task/surrol_bipeg_transfer.yaml
```

第一版配置结构：

```yaml
name: surrol_bipeg_transfer

shape_meta:
  obs:
    point_cloud:
      shape: [1024, 3]
      type: point_cloud
    agent_pos:
      shape: [14]
      type: low_dim
  action:
    shape: [14]

dataset:
  _target_: diffusion_policy_3d.dataset.surrol_bipeg_dataset.SurrolBipegDataset
  npz_path: collected_data/bipeg_transfer_dp3_pointcloud.npz
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: ${training.seed}
  val_ratio: 0.2

env_runner:
  _target_: diffusion_policy_3d.env_runner.surrol_bipeg_runner.SurrolBipegRunner
  env_id: BiPegTransfer-v4
  action_mode: rpy
  eval_episodes: 20
  max_steps: 200
```

训练脚本会覆盖 `task.env_runner=null`，只做 offline imitation training。
评估脚本使用这里配置的 SurRoL online runner 加载 checkpoint 做 rollout。

## 4. Offline Training

训练脚本：

```text
Methods/dp3-surrol/3D-Diffusion-Policy/scripts/train_surrol_bipeg_transfer.sh
```

建议默认不要直接用原版 255M 参数 DP3。第一版用 tiny/small override：

```bash
cd Methods/dp3-surrol/3D-Diffusion-Policy/3D-Diffusion-Policy

python3 train.py --config-name=dp3.yaml \
  task=surrol_bipeg_transfer \
  training.debug=False \
  training.num_epochs=50 \
  training.seed=0 \
  training.device=cuda:0 \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  checkpoint.save_ckpt=True \
  checkpoint.save_last_ckpt=True \
  training.checkpoint_every=1 \
  logging.mode=disabled \
  policy.down_dims=[128,256,512] \
  policy.encoder_output_dim=32 \
  policy.diffusion_step_embed_dim=64 \
  exp_name=surrol_bipeg_transfer-dp3-tiny50 \
  hydra.run.dir=data/outputs/surrol_bipeg_transfer-dp3-tiny50_seed0
```

训练完成后必须得到：

```text
data/outputs/surrol_bipeg_transfer-dp3-tiny50_seed0/checkpoints/latest.ckpt
```

如果没有 checkpoint，说明 `checkpoint.save_ckpt=True` 没打开。

## 5. SurRoL Online Observation Adapter

在线评估时不能读 pickle；必须从当前 SurRoL 环境实时生成和训练一致的输入。

每个 timestep：

1. `obs = env.reset()` 或 `obs, reward, done, info = env.step(action)`。
2. 从 `obs` 读取：
   - `obs["images"]["rgb1"]`
   - `obs["images"]["depth"]`
   - `obs["images"]["mask"]`
   - `obs["qpos"]`
3. 复用 converter 的 `depth_to_camera_points()` 和 `sample_points()`。
4. 如果训练时是 `mask_mode=target`，在线 mask 应使用：
   - `mask == 1`
   - `mask == 4`
   - `mask == env.unwrapped.target_id`
5. 得到：
   - `point_cloud`: `(1024, 3)`
   - `agent_pos`: `(14,)`

第一版可以使用模拟器 segmentation mask。这不是现实部署设置，但和当前
`collected_data` 的 `target` mask 数据分布一致。若要避免 oracle segmentation，
应重新生成 `--mask-mode all` 数据并训练对应模型。

## 6. SurRoL Env Runner

online runner：

```text
diffusion_policy_3d/env_runner/surrol_bipeg_runner.py
```

runner 行为：

```text
for episode in eval_episodes:
    env = gym.make("BiPegTransfer-v4", render_mode="rgb_array", action_mode="rpy")
    obs = env.reset()
    obs_deque = first observation repeated n_obs_steps times
    policy.reset()

    while not done and step < max_steps:
        obs_dict = {
            "point_cloud": stack(obs_deque.point_cloud)[None],
            "agent_pos": stack(obs_deque.agent_pos)[None],
        }
        action_seq = policy.predict_action(obs_dict)["action"][0]

        for action in action_seq[:n_action_steps]:
            obs, reward, done, info = env.step(action)
            append current point_cloud/agent_pos to obs_deque
            record rgb frame, reward, success, action
            if done or info["is_success"] > 0:
                break
```

输出指标：

- `test_mean_score`: success rate。
- `mean_success_rate`: success rate。
- `mean_reward`: average episode reward。
- `mean_episode_length`: average executed timesteps。

输出文件：

```text
data/outputs/<run_name>/eval_outputs/
  results.json
  episodes.csv
  videos/episode_000.mp4
  videos/episode_001.mp4
```

## 7. Eval Script

eval 脚本：

```text
Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh
```

目标命令形式：

```bash
cd Methods/dp3-surrol/3D-Diffusion-Policy/3D-Diffusion-Policy

python3 eval.py --config-name=dp3.yaml \
  task=surrol_bipeg_transfer \
  training.device=cuda:0 \
  task.env_runner.eval_episodes=20 \
  task.env_runner.max_steps=200 \
  task.env_runner.output_video=True \
  hydra.run.dir=data/outputs/surrol_bipeg_transfer-dp3-tiny50_seed0
```

实际入口：

```bash
MODEL_SIZE=tiny EVAL_EPISODES=20 OUTPUT_VIDEO=True \
  bash Methods/dp3-surrol/3D-Diffusion-Policy/scripts/eval_surrol_bipeg_transfer.sh 0 0 train50
```

## 8. 推荐实施顺序

按以下顺序实现，不要跳步：

1. **确认数据**：运行 converter 和 `inspect_dataset_batch.py`，确认 NPZ shape。
2. **移植 dataset**：把 SurRoL dataset 放进 DP3 包内，Hydra 能 instantiate。
3. **移植 task config**：`task=surrol_bipeg_transfer` 能启动 offline train。
4. **短训保存 checkpoint**：tiny 模型跑 50 epoch，保存 `latest.ckpt`。
5. **实现 online observation adapter**：单独测试 env reset 后能生成 `(1024,3)` 点云。
6. **实现 runner**：用未训练或短训 policy 跑 1 episode，不要求成功，只要求不崩。
7. **正式 eval**：跑 20 episodes，输出 `results.json/videos/episodes.csv`。
8. **扩展实验**：比较 tiny/small/base、`target/all` mask、`1024/2048` points。

## 9. 成功标准

第一阶段成功：

- DP3 能从 NPZ 训练。
- loss 能稳定下降。
- 能保存 `latest.ckpt`。

第二阶段成功：

- `BiPegTransfer-v4` 在线 rollout 不崩。
- policy 输出 14-D action。
- `env.step(action)` 能执行完整 episode。
- 能输出 `results.json` 和 rollout video。

第三阶段成功：

- success rate 明显高于随机策略。
- 不同 seed 下结果可复现。
- 训练和 eval 的点云生成参数完全一致。

## 10. 关键风险

- **训练/eval mask 不一致**：训练用 `target`，eval 也必须用同样 target mask。
- **动作语义不一致**：训练用 `actions`，eval 必须直接 `env.step(action)`。
- **点云坐标系不一致**：当前是 ECM camera frame，online 也必须保持 camera frame。
- **normalizer 不一致**：训练保存的 normalizer 必须随 checkpoint 使用。
- **没有 checkpoint**：正式训练必须打开 `checkpoint.save_ckpt=True`。
- **只看 loss 不够**：最终指标必须来自 SurRoL online rollout success rate。
