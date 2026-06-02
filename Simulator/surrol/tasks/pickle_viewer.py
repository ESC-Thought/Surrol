import pickle

# 加载文件
file_path = '/home/ubuntu/SurRoL/SurRoL_Mygit/IROS_SurRoL/surrol/data/image_action_pairs_0'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 查看数据内容
images = data['images']
actions = data['actions']

print(f"Number of images: {len(images)}")
print(f"Number of actions: {len(actions)}")

# 查看第一个图像和动作数据
print(f"First left image shape: {images[0][0].shape}")  # 可能是 RGB 图像的 numpy 数组
print(f"First right image shape: {images[0][1].shape}")  # 可能是 RGB 图像的 numpy 数组
print(f"First action: {actions[0]}")
