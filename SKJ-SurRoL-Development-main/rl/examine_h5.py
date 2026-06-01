import h5py
import sys
import os

def print_h5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

h5_path = "/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/eval/whole_policy/0625-stereo-200_correction/correction_data/episode_000/data.h5"

try:
    if not os.path.exists(h5_path):
        print(f"Error: File not found at {h5_path}")
        sys.exit(1)
        
    print(f"\nAttempting to open: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        print("\nHDF5 File Structure:")
        print("===================")
        f.visititems(print_h5_structure)
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1) 