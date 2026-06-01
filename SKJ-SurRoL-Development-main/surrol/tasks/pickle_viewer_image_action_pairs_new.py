import pickle

# 指定文件路径
file_path = '/home/skylar/SurRoL/IROS_SurRoL/rl/act-main-3/data/0218-bipeg-target/image_action_qpos_1.pkl'

# 加载数据
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 提取路径点信息
images = data['images']  # 字典，键为路径点
actions = data['actions']  # 字典，键为路径点
qpos = data['qpos']  # 字典，键为路径点
dqpos = data['dq_actions']  # 字典，键为路径点
masks = data['masks_target']  # 字典，键为路径点
depths = data['depths']  # 字典，键为路径点

# 打印总体信息
print(f"Waypoints: {list(images.keys())}")  # 路径点列表
for waypoint in images.keys():
    print(f"\nWaypoint: {waypoint}")
    print(f"  Number of image pairs: {len(images[waypoint])}")
    print(f"  Number of actions: {len(actions[waypoint])}")
    print(f"  Number of qpos: {len(qpos[waypoint])}")
    print(f"  Number of dqpos: {len(dqpos[waypoint])}")
    print(f"  Number of masks: {len(masks[waypoint])}")
    print(f"  Number of depths: {len(depths[waypoint])}")

    # 检查第一个图像和动作
    if len(images[waypoint]) > 0:
        first_image_pair = images[waypoint][0]
        first_action = actions[waypoint][0]
        first_qpos = qpos[waypoint][0]
        first_dqpos = dqpos[waypoint][0]
        first_mask = masks[waypoint][0]
        first_depth = depths[waypoint][0]

        print(f"  First left image shape: {first_image_pair[0].shape}")  # 左图
        print(f"  First right image shape: {first_image_pair[1].shape}")  # 右图
        print(f"  First action: {first_action}")
        print(f"  First qpos: {first_qpos}")
        print(f"  First dqpos: {first_dqpos}")
        print(f"  First mask shape: {first_mask.shape}")
        print(f"  First depth shape: {first_depth.shape}")

# 查看第 n 个数据（可选）
n = 0
for waypoint in images.keys():
    if len(images[waypoint]) > n:
        print(f"\nWaypoint: {waypoint} - Data at index {n}")
        print(f"  Left image shape: {images[waypoint][n][0].shape}")
        print(f"  Right image shape: {images[waypoint][n][1].shape}")
        print(f"  Action: {actions[waypoint][n]}")
        print(f"  Qpos: {qpos[waypoint][n]}")
        print(f"  DQpos: {dqpos[waypoint][n]}")
    else:
        print(f"\nWaypoint: {waypoint} - Index {n} out of range!")
