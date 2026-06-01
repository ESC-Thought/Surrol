import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from constants import waypoints
from constants import SIM_TASK_CONFIGS
import IPython
e = IPython.embed
import pickle

# camera_names = SIM_TASK_CONFIGS['needle_pick']['camera_names']
# images_input = SIM_TASK_CONFIGS['needle_pick']['images_input']
# waypoints = ['0','1','2','3']
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, images_input, norm_stats, waypoints, action_mode='dq', max_episode_len=200):
        super(EpisodicDataset).__init__()
        self.max_episode_len = max_episode_len  # Store max episode length
        # Pre-load all data into memory
        self.data_cache = {}
        for episode_id in episode_ids:
            dataset_path = os.path.join(dataset_dir, f'image_action_qpos_{episode_id}.pkl')
            with open(dataset_path, 'rb') as f:
                self.data_cache[episode_id] = pickle.load(f)
        # Store other parameters
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.images_input = images_input
        self.norm_stats = norm_stats
        self.waypoints = waypoints
        self.action_mode = action_mode
        self.is_sim = None
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        """
            waypoint_images = [
                [camera1_image1, camera2_image1],  # Images at time step 1
                [camera1_image2, camera2_image2],  # Images at time step 2
                ...
            ]
            all_images = [ 
                [camera1_image1, camera2_image1],
                [camera1_image2, camera2_image2],
                ...
                [camera1_imageN, camera2_imageN]
            ]  # all images from different waypoints
        """
        sample_full_episode = False #TODO # hardcode  # 是否采样整个 episode，设置为 False 表示只采样一帧

        episode_id = self.episode_ids[index]    # 根据索引获取 episode ID
        # print(f"Loading episode {episode_id}...")
        data = self.data_cache[episode_id]
        
        all_images, all_actions, all_qpos, all_masks, all_depths = [], [], [], [], []


        # original_action_shape = data['actions'].shape
        for waypoint in self.waypoints:
            if waypoint in data['actions']:
                waypoint_images = data['images'][waypoint]
                # print("Type of waypoint_images:", type(waypoint_images))
                # if isinstance(waypoint_images, np.ndarray):  # If it's a NumPy array
                #     print("Shape of waypoint_images:", waypoint_images.shape)
                # elif isinstance(waypoint_images, list):  # If it's a list
                #     print("Length of waypoint_images:", len(waypoint_images))
                #     print("Type of first element in waypoint_images:", type(waypoint_images[0]))
                #     if isinstance(waypoint_images[0], np.ndarray):
                #         print("Shape of first element in waypoint_images:", waypoint_images[0].shape)
                #     else:
                #         print("Number of elements in Image 0:", len(waypoint_images[0]))
                #         for j, subelement in enumerate(waypoint_images[0]):
                #             print("aaaaaaaaaaaaa:", subelement.shape)  (256,256,3)

                if self.action_mode != 'dq':
                    waypoint_actions = np.array(data['actions'][waypoint])
                elif self.action_mode == 'dq':
                    waypoint_actions = np.array(data['dq_actions'][waypoint])
                else:
                    raise ValueError('Invalid action mode')
                waypoint_qpos = np.array(data['qpos'][waypoint])
                waypoint_masks = np.array(data['masks_target'][waypoint])  # TODO-CAM
                # waypoint_masks = np.array(data['masks'][waypoint])
                # waypoint_depths = np.array(data['depths'][waypoint])

                # 合并路径点数据
                all_images.extend(waypoint_images)
                all_actions.extend(waypoint_actions)
                all_qpos.extend(waypoint_qpos)
                all_masks.extend(waypoint_masks)
                # all_depths.extend(waypoint_depths)
        # print("all_actions:",all_actions)
        # episode_len = len(data['actions'])
        episode_len = len(all_actions)

        # Ensure all episodes have exactly the same length
        episode_len = self.max_episode_len

        if sample_full_episode:
            start_ts = 0
        else:
            # Make sure we don't go out of bounds when selecting start_ts
            max_start = min(len(all_actions), episode_len)
            start_ts = np.random.choice(max_start) if max_start > 0 else 0
        # 提取图像、动作和关节状态  
        # Image tensor shape: (1000, 2, 256, 256, 3)
        # images = np.array(data['images'])  # 假设图像数据的形状为 (N, H, W, C)
        image_dict = dict()
        # print("IMAGE!!!:",all_images[1])
        # print("IMAGE 0:",np.array(all_images)[:,0,:,:,:])
        # print("IMAGE 1",np.array(all_images)[:,1,:,:,:])
        # for i in range(2): 
        for i in range(len(self.camera_names)):
            # print("camera_names:", camera_names)
            # print("all images:", all_images)
            # image_dict[i] = np.array(data['images'])[:, i, :, :, :][start_ts]  # 提取相机的图像 (N, 256, 256, 3)
            image_dict[i] = np.array(all_images)[:, i, :, :, :][start_ts]  # 提取相机的图像 (N, 256, 256, 3)
            cnt = i
        if self.images_input is not None: # TODO edit this if there is a different image input (sky)
            # print(f"waypoint_depths size: {waypoint_depths.shape if isinstance(waypoint_depths, np.ndarray) else len(waypoint_depths)}")
            # print(f"start_ts: {start_ts}")
            # print(f"cnt: {cnt}")

            # image_dict[cnt+1] = waypoint_depths[start_ts]
            # image_dict[cnt+2] = waypoint_masks[start_ts]
            if 'mask' in self.images_input:
                image_dict[cnt+1] = all_masks[start_ts]
            if 'depth' in self.images_input:
                image_dict[cnt+2] = all_depths[start_ts]

        actions = np.array(all_actions[start_ts:])
        action_len = min(len(actions), episode_len)  # Limit action length

        qpos = np.array(all_qpos[start_ts])
        # print(f"Image tensor shape: {images.shape}")
        # padded_action = np.zeros_like(data['actions'], dtype=np.float32)    # 初始化填充动作
        
        padded_action = np.zeros((episode_len, len(all_actions[0])), dtype=np.float32)
        padded_action[:action_len] = actions[:action_len]  # Only copy up to action_len
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        all_cam_images = []
        #########################################################
        #########################################################
        ## edit for only rgb1 input
        camera_names_new = self.camera_names + self.images_input if self.images_input is not None else self.camera_names        
        for i in range(len(camera_names_new)):
            img = image_dict[i]
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)  
                img = np.repeat(img, 3, axis=-1)    
            all_cam_images.append(img)
        all_cam_images = np.stack(all_cam_images, axis=0)
        # print(f"all:,{all_cam_images.shape}")
        #########################################################
        #########################################################
        # img = image_dict[0]  # Only take the first camera (rgb1)
        # all_cam_images = np.expand_dims(img, axis=0)  # Add dimension to match expected shape
        #########################################################
        #########################################################
        images = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        # print(f"shape of image_data:,{images.shape}")
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', images)
        # normalize image and change dtype to float
        image_data = image_data / 255.0   # 归一化到[0,1]
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        """
        print(f"Index {index}:")
        print(f" u Image shape: {image_data.shape}")
        print(f" u  Qpos shape: {qpos_data.shape}")
        print(f" u Actions shape: {action_data.shape}")
        print(f" u Is_pad shape: {is_pad.shape}")
         u Image shape: torch.Size([3, 3, 256, 256])
         u  Qpos shape: torch.Size([6])
         u Actions shape: torch.Size([50, 6])
         u Is_pad shape: torch.Size([50])
        """
        return image_data, qpos_data, action_data, is_pad
        

def get_norm_stats(dataset_dir, num_episodes, waypoints, action_mode='dq'):
    """
    计算归一化参数（均值和标准差）用于对动作和关节状态（qpos）进行标准化。
    数据集格式为 `image_action_qpos_{episode_idx}.pkl`。
    """
    all_qpos_data = []
    all_action_data = []

    for episode_idx in range(num_episodes):
        file_path = os.path.join(dataset_dir, f'image_action_qpos_{episode_idx}.pkl')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)  # 加载单个 episode 的数据

        for waypoint in waypoints:
            if waypoint in data['qpos']:
                # 提取 qpos 和动作数据
                qpos = np.array(data['qpos'][waypoint])
                if action_mode != 'dq':
                    actions = np.array(data['actions'][waypoint])
                elif action_mode == 'dq':
                    actions = np.array(data['dq_actions'][waypoint])

                all_qpos_data.append(qpos)
                all_action_data.append(actions)
            else:
                print(f"waypoint {waypoint}is not in data")
    # 合并所有 episode 的数据
    all_qpos_data = np.vstack(all_qpos_data)
    all_action_data = np.vstack(all_action_data)

    # 计算均值和标准差
    action_mean = all_action_data.mean(axis=0)
    action_std = np.clip(all_action_data.std(axis=0), 1e-2, np.inf)  # 防止标准差过小导致归一化失效

    qpos_mean = all_qpos_data.mean(axis=0)
    qpos_std = np.clip(all_qpos_data.std(axis=0), 1e-2, np.inf)  # 同样处理标准差过小的情况

    stats = {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "example_qpos": all_qpos_data[0]  # 提供一个示例 qpos 数据（可选）
    }

    return stats



# def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
#     print(f'\nData from: {dataset_dir}\n')
#     # obtain train test split
#     train_ratio = 0.8
#     shuffled_indices = np.random.permutation(num_episodes)
#     train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
#     val_indices = shuffled_indices[int(train_ratio * num_episodes):]

#     # obtain normalization stats for qpos and action
    # norm_stats = get_norm_stats(dataset_dir, num_episodes)

#     # construct dataset and dataloader
#     train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
#     val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

#     return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

# sky
def load_data(dataset_dir, num_episodes, camera_names, images_input, batch_size_train, batch_size_val, waypoints=['0'], action_mode='dq'):
    """
    加载 image_action_qpos 数据集，并构建训练和验证数据加载器。
    """
    print(f'\nLoading data from: {dataset_dir}\n')

    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # 获取数据集的归一化统计信息
    norm_stats = get_norm_stats(dataset_dir, num_episodes, waypoints=waypoints, action_mode=action_mode)

    # 构建训练和验证数据集
    # train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names=['top'], norm_stats=norm_stats,waypoints=waypoints)
    # val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names=['top'], norm_stats=norm_stats,waypoints=waypoints)
    train_dataset = EpisodicDataset(
        train_indices, 
        dataset_dir, 
        camera_names, 
        images_input, 
        norm_stats=norm_stats,
        waypoints=waypoints, 
        action_mode=action_mode,
        max_episode_len=200
    )
    
    val_dataset = EpisodicDataset(
        val_indices, 
        dataset_dir, 
        camera_names, 
        images_input,
        norm_stats=norm_stats,
        waypoints=waypoints, 
        action_mode=action_mode,
        max_episode_len=200
    )


    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=4,  # Increase from 1
        prefetch_factor=2,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True, 
        pin_memory=True,
        num_workers=4,  # Increased from 1
        prefetch_factor=2,  # Increased from 1
        persistent_workers=True
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
