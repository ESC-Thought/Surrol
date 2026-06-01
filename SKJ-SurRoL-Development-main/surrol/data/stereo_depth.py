import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
from easydict import EasyDict as edict

# Add IGEV to path
sys.path.append('/research/d1/gds/kjshi/IROS_SurRoL/IGEV/core')
sys.path.append('/research/d1/gds/kjshi/IROS_SurRoL/IGEV')
from igev_stereo import IGEVStereo
from IGEV.core.utils.utils import InputPadder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_depth_model(checkpoint_path='/research/d1/gds/kjshi/IROS_SurRoL/pretrained_models/sceneflow.pth'):
    """Load IGEV stereo depth estimation model"""
    args = edict()
    args.restore_ckpt = checkpoint_path
    args.mixed_precision = False
    args.valid_iters = 32
    args.hidden_dims = [128]*3
    args.corr_implementation = "reg"
    args.shared_backbone = False
    args.corr_levels = 2
    args.corr_radius = 4
    args.n_downsample = 2
    args.slow_fast_gru = False
    args.n_gru_layers = 3
    args.max_disp = 192

    model = IGEVStereo(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.module
    model.to(device)
    model.eval()
    return model

def convert_disparity_to_depth(disparity, baseline, focal_length):
    """Convert disparity to depth"""
    depth = baseline * focal_length / (disparity + 1e-6)
    return depth

def get_depth_from_stereo(model, left_img, right_img, baseline, focal_length):
    """Get depth from stereo images"""
    # Resize images if needed
    if left_img.shape[:2] != (256, 256):
        left_img = cv2.resize(left_img, (256, 256))
        right_img = cv2.resize(right_img, (256, 256))
    
    # Convert to float and normalize
    left_img = left_img.astype(np.float32) / 255.0
    right_img = right_img.astype(np.float32) / 255.0
    
    # Convert to torch tensors
    left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).float().to(device).unsqueeze(0)
    right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).float().to(device).unsqueeze(0)

    with torch.no_grad():
        padder = InputPadder(left_tensor.shape, divis_by=32)
        image1, image2 = padder.pad(left_tensor, right_tensor)
        disp = model(image1, image2, iters=32, test_mode=True)
        disp = padder.unpad(disp).squeeze()
        disp = disp.cpu().numpy()
    
    # Convert disparity to depth
    depth = convert_disparity_to_depth(disp, baseline, focal_length)
    
    return depth, disp

# Camera parameters (adjust these according to your stereo camera setup)
RENDER_HEIGHT = 256
RENDER_WIDTH = 256
FoV = 20  # in degrees (adjust this to match your camera)
baseline = -0.004214  # distance between cameras (adjust this to your setup)
focal_length = (RENDER_WIDTH / 2) / np.tan(np.radians(FoV / 2))

# Load model
model = load_depth_model()

# Example usage with your stereo images:
left_img = cv2.imread('path/to/left_image.jpg')  # Replace with your left image path
right_img = cv2.imread('path/to/right_image.jpg')  # Replace with your right image path

# Get depth
depth_map, disparity = get_depth_from_stereo(model, left_img, right_img, baseline, focal_length)

# Visualize
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
plt.title('Left Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(disparity, cmap='jet')
plt.colorbar(label='Disparity')
plt.title('Disparity Map')
plt.axis('off')

plt.subplot(133)
plt.imshow(depth_map, cmap='jet')
plt.colorbar(label='Depth (m)')
plt.title('Estimated Depth')
plt.axis('off')

plt.tight_layout()
plt.savefig('stereo_depth_result.png')
plt.close()

# Save depth map
np.save('depth_map.npy', depth_map)

print("Depth estimation completed!") 