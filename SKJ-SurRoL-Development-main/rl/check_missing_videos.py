import os
import glob

def check_missing_videos(base_dir):
    """
    Check all episode directories for missing MP4 files.
    
    Args:
        base_dir (str): Base directory containing correction_data folder
    """
    # Path to correction_data directory
    correction_data_dir = os.path.join(base_dir, 'correction_data')
    
    if not os.path.exists(correction_data_dir):
        print(f"Error: Directory {correction_data_dir} does not exist!")
        return
    
    # Get all episode directories
    episode_dirs = glob.glob(os.path.join(correction_data_dir, 'episode_*'))
    episode_dirs.sort()  # Sort to process in order
    
    missing_videos = []
    complete_episodes = []
    
    print("\nChecking episodes for missing videos...")
    print("=" * 50)
    
    for episode_dir in episode_dirs:
        episode_num = int(os.path.basename(episode_dir).split('_')[1])
        
        # Check for h5 and mp4 files
        h5_file = os.path.join(episode_dir, 'data.h5')
        mp4_file = os.path.join(episode_dir, 'episode.mp4')
        
        if os.path.exists(h5_file) and not os.path.exists(mp4_file):
            missing_videos.append(episode_num)
        elif os.path.exists(h5_file) and os.path.exists(mp4_file):
            complete_episodes.append(episode_num)
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    print(f"Total episodes checked: {len(episode_dirs)}")
    print(f"Complete episodes: {len(complete_episodes)}")
    print(f"Episodes missing videos: {len(missing_videos)}")
    
    if missing_videos:
        print("\nEpisodes missing MP4 files:")
        print("-" * 50)
        for episode_num in missing_videos:
            print(f"Episode {episode_num:03d}")
    
    return missing_videos, complete_episodes

if __name__ == "__main__":
    # Base directory path - adjust this to match your setup
    base_dir = os.path.expanduser('~/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/eval/whole_policy/0625-stereo-200_correction/')
    
    missing_videos, complete_episodes = check_missing_videos(base_dir) 