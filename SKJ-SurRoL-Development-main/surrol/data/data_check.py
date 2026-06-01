import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
import math

# Add surrol to Python path if needed
surrol_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if surrol_path not in sys.path:
    sys.path.append(surrol_path)

# Camera parameters
RENDER_HEIGHT = 256
RENDER_WIDTH = 256
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
# with open('/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/collected_data/needle_pick/image_action_qpos_1.pkl', 'rb') as file:
with open('/home/ztt/Desktop/KejianShi/IROS_SurRoL/collected_data/peg_transfer_stereo0729_final/image_action_qpos_0.pkl', 'rb') as file:
    data = pickle.load(file)

# images = np.array(data['images']['0'])[:, 0, :, :, :] # Extract images from the data
# images2 = np.array(data['images']['0'])[:, 1, :, :, :]
images = np.array(data['depths']['0'][0])  # Extract images from the data
print(images.max(), images.min())
images = (images - images.min()) / (images.max() - images.min())
print(images.max(), images.min())
print(images.shape)
plt.imshow(images)
plt.show()
exit()
masks_target = np.array(data['masks_target']['2'])  # Extract actions from the data
print(f"Type of masks_target: {type(masks_target)}")
# print(f"Shape of masks_target: {(masks_target)}")

# plt.figure(figsize=(15, 5 * rows))
num_images = images.shape[0]

cols = 2  # Number of columns for plotting
rows = (num_images*2 // cols) + (num_images*2 % cols > 0)  # Calculate number of rows

plt.figure(figsize=(15, 5 * rows))
for i, image in enumerate(images):
    # mask = masks_target[i]
    # mask_vis = (mask * 255).astype(np.uint8)
    # # 叠加 mask 到原图像（使用红色通道）
    # overlay = image.copy()
    # overlay[:, :, 0] = np.maximum(overlay[:, :, 0], mask_vis)  # 在红色通道叠加 mask
    img2 = images2[i]
    # 显示原始图像
    plt.subplot(rows, cols, 2 * i + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Image {i + 1}')

    # 显示 mask
    plt.subplot(rows, cols, 2 * i + 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title(f'Image2 {i + 1}')
    # # print(image.dtype)
    # Save the plot
    # plt.subplot(rows, cols, i + 1)  # Create subplot for each image
    # # plt.imshow(np.array((image==6)|(image==1)), cmap='gray')  # Adjust cmap as necessary
    # plt.imshow(image, cmap='gray')  # Adjust cmap as necessary
    # plt.axis('off')  # Hide axes
    # plt.title(f'Image {i + 1}')  # Title for each image

plt.tight_layout()

plt.show()
plt.savefig('visualization.png')
