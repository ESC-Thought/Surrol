import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
import cv2
import torch
import argparse
from pathlib import Path
import json
sys.path.append('src')
sys.path.append('./')
from src.gd.perception import *
from unpack_data import unpack_data
import math
from scipy.spatial.transform import Rotation
import pickle
import numpy as np
def calculate_residual_optical_flow(optical_flow):
    """calculate the residual optical flow from the optical flow imageB

    Args:
        optical_flow (_type_): _description_
    """
    original_pixel_image = np.indices((360, 640), dtype=np.float32).transpose(1,2,0)
    residual_optical_flow = optical_flow - original_pixel_image
    return residual_optical_flow

def write_txt(vertices, file_path, overwrite=True):
    """avoid writing in a loop by creating a template for the entire array, and format at once"""
    if overwrite:
        with open(file_path, 'w') as obj_file:
            obj_file.write('')
    with open(file_path, 'a') as test_file:
        temp = '%s %s %s\n' * len(vertices)
        test_file.write(temp % tuple(vertices.ravel()))


def depth2xyz(depth_img, camera_params):
    # scale camera parameters
    h, w = depth_img.shape
    scale_x = w / camera_params['xres']
    scale_y = h / camera_params['yres']
    fx = camera_params['fx'] * scale_x
    fy = camera_params['fy'] * scale_y
    x_offset = camera_params['cx'] * scale_x
    y_offset = camera_params['cy'] * scale_y
    indices = torch.from_numpy(np.indices((h, w), dtype=np.float32).transpose(1,2,0))
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = torch.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def xyz2depth(cam_img, camera_params):
    '''
    cam_img: h,w,3
    '''
    
    fx = camera_params['fx']
    fy = camera_params['fy']
    x_offset = camera_params['cx']
    y_offset = camera_params['cy']

    u = fx * (cam_img[...,  0]/ cam_img[...,2]) + x_offset
    v = fy * (cam_img[..., 1]/ cam_img[ ...,2]) + y_offset
    pixel_img = torch.stack([v, u], axis=-1)
    return pixel_img
    
    
def cam2world(cam_img, rotation, translation):
    '''
    cam_img: H,W,3 or H*W, 3
    rotation and translations are from camera coordinates to world coordinates
    rotation: 3*3
    translation: 3*1

    out: shape HW, 3
    '''
    cam_img = cam_img.transpose(1,0)
    out = rotation.float() @ cam_img.float() + translation.reshape(3,1).float()

    return out.transpose(1,0)

def world2cam(world_img, rotation, translation):
    '''
    world_img: H*W*3
    rotation and translations are from camera to world, need to inverse here
    '''
    world_img = world_img.transpose(1,0)
    out = torch.linalg.inv(rotation).float() @ (world_img.float() - translation.reshape(3,1).float())
    return out.transpose(1,0)

def pixel2xy1(pixel_img, camera_params):
    # scale camera parameters
    h, w = pixel_img.shape[:2]
    scale_x = w / camera_params['xres']
    scale_y = h / camera_params['yres']
    fx = camera_params['fx'] * scale_x
    fy = camera_params['fy'] * scale_y
    x_offset = camera_params['cx'] * scale_x
    y_offset = camera_params['cy'] * scale_y

    z_e = torch.ones((h, w))
    x_e = (pixel_img[..., 1] - x_offset) / fx
    y_e = (pixel_img[..., 0] - y_offset) / fy
    xyz_img = torch.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img

def recover_depth(of1_data, camera_rotA, camera_rotB, q1, q2, camera_params, maskA=None):
    h, w, _ = of1_data.shape
    original_pixel = torch.from_numpy(np.indices((h, w), dtype=np.float32).transpose(1, 2, 0))
    of1_data = torch.from_numpy(of1_data)
    of1_data += original_pixel

    if maskA is not None:
        xy1A = pixel2xy1(of1_data, camera_params)
        xy1A = xy1A.reshape(-1, 3)[maskA.cpu().numpy().reshape(-1,), :]
        xy1B = pixel2xy1(original_pixel, camera_params)
        xy1B = xy1B.reshape(-1, 3)[maskA.cpu().numpy().reshape(-1,), :]
    else:    
        # xy1A = pixel2xy1(of1_data, camera_params)
        # xy1A = xy1A.reshape(-1, 3)
        # xy1B = pixel2xy1(original_pixel, camera_params)
        # xy1B = xy1B.reshape(-1, 3)
        
        xy1A = pixel2xy1(original_pixel, camera_params)
        xy1A = xy1A.reshape(-1, 3)
        xy1B = pixel2xy1(of1_data, camera_params)
        xy1B = xy1B.reshape(-1, 3)
    
    dir1 = camera_rotA.float() @ xy1A.transpose(1, 0).float()  # 3, -1
    dir2 = camera_rotB.float() @ xy1B.transpose(1, 0).float()  # 3, -1

    dir1 = dir1.reshape((3, -1, 1)).transpose(1,0)
    dir2 = dir2.reshape((3, -1, 1)).transpose(1,0)
    

    q12 = (q1 - q2).reshape(3, 1)
    q21 = (q2 - q1).reshape(3, 1)
    theta = torch.acos(((torch.einsum('dij,dij->d', dir1, dir2) ** 2)/(torch.einsum('dij,dij->d', dir1, dir1) * torch.einsum('dij,dij->d', dir2, dir2)))**0.5)
    mask = (theta > 5e-3) #(batch,)

    dir1 = dir1[mask]
    dir2 = dir2[mask]

    inverse_scalar = 1 / ((torch.einsum('dij,dij->d', dir1, dir1) * torch.einsum('dij,dij->d', dir2, dir2) - torch.einsum('dij,dij->d', dir1, dir2) ** 2))

    # print((torch.einsum('dij,dij->d', dir1, dir1) * torch.einsum('dij,dij->d', dir2, dir2) - torch.einsum('dij,dij->d', dir1, dir2) ** 2))
    # exit()
    mat1_11 = torch.einsum('dij,dij->d', dir2, dir2)
    mat1_12 = torch.einsum('dij,dij->d', dir1, dir2)
    mat1_21 = torch.einsum('dij,dij->d', dir2, dir1)
    mat1_22 = torch.einsum('dij,dij->d', dir1, dir1)

    mat1_1 = torch.stack([mat1_11, mat1_12], dim=1)
    mat1_2 = torch.stack([mat1_21, mat1_22], dim=1)
    mat1 = torch.stack([mat1_1, mat1_2], dim=1)
    mat2_1 = torch.einsum('dij,ij->d', dir1.float(), q21.float())
    mat2_2 = torch.einsum('dij,ij->d', dir2.float(), q12.float())
    mat2 = torch.stack([mat2_1, mat2_2], dim=1)

    # mat2 = torch.stack([(torch.einsum('dij,ji->d', dir1.transpose(2,1), q21)),\
    #                     (torch.einsum('dij,ji->d', dir2.transpose(2,1), q12))], dim=1)
    
    result_mat = torch.einsum('dij,dj->di', mat1, mat2)

    result = inverse_scalar.reshape((-1, 1)) * result_mat

    p = q1.reshape(3, 1) + result[:, 0][:,None].view(-1, 1, 1) * dir1
    q = q2.reshape(3, 1) + result[:, 1][:,None].view(-1, 1, 1) * dir2

    result_list = (p + (q - p) / 2).detach().numpy().tolist()

    return result_list, mask