import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
import os
import sys
import math

# Add surrol to Python path if needed
surrol_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if surrol_path not in sys.path:
    sys.path.append(surrol_path)

# Camera parameters
RENDER_HEIGHT = 480
RENDER_WIDTH = 640
FoV = 60  # in degrees

# Calculate focal length from FoV
focal_length = (RENDER_WIDTH / 2) / math.tan(math.radians(FoV / 2))

# Camera intrinsic matrix
CAMERA_INTRINSIC_MATRIX = np.array([
    [focal_length, 0., RENDER_WIDTH/2],
    [0., focal_length, RENDER_HEIGHT/2],
    [0., 0., 1.]
])

# Extract camera intrinsics from the matrix
fx = CAMERA_INTRINSIC_MATRIX[0, 0]
fy = CAMERA_INTRINSIC_MATRIX[1, 1]
cx = CAMERA_INTRINSIC_MATRIX[0, 2]
cy = CAMERA_INTRINSIC_MATRIX[1, 2]

print(f"Camera Intrinsics:")
print(f"fx: {fx}, fy: {fy}")
print(f"cx: {cx}, cy: {cy}")

# Load the pickle file
with open('/research/d1/gds/kjshi/IROS_SurRoL/collected_data/bi_peg_transfer_foveal/image_action_qpos_0.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract the foveal images
foveal_block = np.array(data['foveal_block']['3'])  # Shape should be (N, H, W, 3)
foveal_tip1 = np.array(data['foveal_tip1']['3'])   # Shape should be (N, H, W, 3)
foveal_tip2 = np.array(data['foveal_tip2']['3'])   # Shape should be (N, H, W, 3)

# Get the number of timesteps
num_timesteps = len(foveal_block)

# Create a figure with 3 rows (one for each foveal view) and num_timesteps columns
plt.figure(figsize=(20, 8))

# Plot foveal images for a few timesteps (e.g., first 5)
num_samples = min(5, num_timesteps)  # Show first 5 timesteps or all if less than 5

for t in range(num_samples):
    # Plot block foveal view
    plt.subplot(3, num_samples, t + 1)
    plt.imshow(foveal_block[t])
    plt.axis('off')
    if t == 0:
        plt.ylabel('Block View', fontsize=12)
    plt.title(f'Step {t}', fontsize=10)
    
    # Plot tip1 foveal view
    plt.subplot(3, num_samples, num_samples + t + 1)
    plt.imshow(foveal_tip1[t])
    plt.axis('off')
    if t == 0:
        plt.ylabel('Tip 1 View', fontsize=12)
    plt.title(f'Step {t}', fontsize=10)
    
    # Plot tip2 foveal view
    plt.subplot(3, num_samples, 2*num_samples + t + 1)
    plt.imshow(foveal_tip2[t])
    plt.axis('off')
    if t == 0:
        plt.ylabel('Tip 2 View', fontsize=12)
    plt.title(f'Step {t}', fontsize=10)

plt.suptitle('Foveal Views Over Time', fontsize=14)
plt.tight_layout()
plt.savefig('foveal_visualization.png', bbox_inches='tight', dpi=300)
plt.show()

# You can also print the shapes to verify the data
print("\nFoveal Image Shapes:")
print(f"Block views shape: {foveal_block.shape}")
print(f"Tip1 views shape: {foveal_tip1.shape}")
print(f"Tip2 views shape: {foveal_tip2.shape}")

# Print pixel positions
recorded_positions = data['recorded_positions']['3']
print("\nRecorded Pixel Positions:")
for t in range(min(5, len(recorded_positions))):
    tip1_pos, tip2_pos, block_pos = recorded_positions[t]
    print(f"Timestep {t}:")
    print(f"  Tip1 pixel: {tip1_pos}")
    print(f"  Tip2 pixel: {tip2_pos}")
    print(f"  Block pixel: {block_pos}")

# def depth_to_pointcloud(depth_image, mask=None, rgb_image=None):
#     """Convert depth image to point cloud with optional masking and color"""
#     height, width = depth_image.shape
    
#     # Create grid of pixel coordinates
#     x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    
#     # Convert depth image to 3D points
#     Z = depth_image
#     X = (x_grid - cx) * Z / fx
#     Y = (y_grid - cy) * Z / fy
    
#     # Stack coordinates
#     points = np.stack([X, Y, Z], axis=-1)
    
#     # Prepare colors if RGB image is provided
#     if rgb_image is not None:
#         # Reshape RGB image to match the points shape
#         colors = rgb_image.reshape(height * width, 3) / 255.0  # Normalize to [0, 1]
    
#     # Apply mask if provided
#     if mask is not None:
#         # Reshape mask to match points shape
#         mask_indices = mask.reshape(-1) > 0
#         valid_points = points.reshape(-1, 3)[mask_indices]
#         if rgb_image is not None:
#             valid_colors = colors[mask_indices]
#     else:
#         valid_mask = Z.reshape(-1) > 0
#         valid_points = points.reshape(-1, 3)[valid_mask]
#         if rgb_image is not None:
#             valid_colors = colors[valid_mask]
    
#     if rgb_image is not None:
#         return valid_points, valid_colors
#     return valid_points

# # After loading the data, convert depths to point clouds and save
# depths = np.array(data['depths']['3'])  # Extract depth images from the data
# print(f"Depth images shape: {depths.shape}")

# # Process each depth image with its corresponding mask
# for i, (depth_image, mask) in enumerate(zip(depths, masks_target)):
#     # Get corresponding RGB image
#     rgb_image = images[i]
    
#     # Convert depth to point cloud using mask and RGB
#     points, colors = depth_to_pointcloud(depth_image, None, rgb_image)
    
#     # Create Open3D point cloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     # Optional: Remove statistical outliers
#     pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
#     # Save point cloud
#     o3d.io.write_point_cloud(f'pointcloud_masked_{i}.ply', pcd)

#     # Optionally visualize the point cloud
#     # o3d.visualization.draw_geometries([pcd])
