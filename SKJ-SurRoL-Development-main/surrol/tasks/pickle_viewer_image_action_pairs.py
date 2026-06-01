import pickle

# 加载文件
# file_path = '/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/1211/image_action_qpos_0.pkl'
# file_path = '/root/autodl-tmp/IROS_SurRoL/rl/act-main-3/data/split_data/image_action_qpos_0.pkl'
file_path = '/root/autodl-tmp/IROS_SurRoL/rl/act-main-3/data/1222exp3-2/image_action_qpos_0.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 提取图像和动作
images = data['images']
actions = data['actions']
qpos = data['qpos']
masks = data['masks']
depths = data['depths']
# 打印总体信息
print(f"Number of image pairs: {len(images)}")
print(f"Number of actions: {len(actions)}")
print(f"Number of qpos: {len(qpos)}")
print(f"Number of masks: {len(masks)}")
print(f"Number of depths: {len(depths)}")
# 检查第一个图像和动作
first_image_pair = images[0]
first_action = actions[0]
first_qpos = qpos[0]
first_mask = masks[0]
first_depths = depths[0]

print(f"First left image shape: {first_image_pair[0].shape}")  # 左图
print(f"First right image shape: {first_image_pair[1].shape}")  # 右图
print(f"First action: {first_action}")
print(f"first qpos {first_qpos}")
print(f"first mask {first_mask}")
print(f"first depth {first_depths}")

# 如果想查看更多数据，例如第 5 个：
n = 5
if n < len(images):
    print(f"Image pair {n} left image shape: {images[n][0].shape}")
    print(f"Image pair {n} right image shape: {images[n][1].shape}")
    print(f"Action {n}: {actions[n]}")
else:
    print(f"Index {n} out of range!")
