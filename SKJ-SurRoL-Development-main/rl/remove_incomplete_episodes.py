import os
import glob
import shutil
from check_missing_videos import check_missing_videos

def remove_incomplete_episodes(base_dir, dry_run=True):
    """
    Remove episode directories that don't have MP4 files.
    
    Args:
        base_dir (str): Base directory containing correction_data folder
        dry_run (bool): If True, only print what would be removed without actually removing
    """
    # First, identify episodes missing videos
    missing_videos, complete_episodes = check_missing_videos(base_dir)
    
    if not missing_videos:
        print("\nNo incomplete episodes found. Nothing to remove.")
        return
    
    # Path to correction_data directory
    correction_data_dir = os.path.join(base_dir, 'correction_data')
    
    # Create backup directory for removed episodes
    backup_dir = os.path.join(base_dir, 'removed_episodes_backup')
    if not dry_run:
        os.makedirs(backup_dir, exist_ok=True)
    
    print("\nProcessing incomplete episodes:")
    print("=" * 50)
    
    for episode_num in missing_videos:
        episode_dir = os.path.join(correction_data_dir, f'episode_{episode_num:03d}')
        backup_episode_dir = os.path.join(backup_dir, f'episode_{episode_num:03d}')
        
        if dry_run:
            print(f"Would remove: {episode_dir}")
            print(f"Would backup to: {backup_episode_dir}")
        else:
            print(f"Moving {episode_dir} to backup...")
            try:
                # Move to backup instead of deleting
                shutil.move(episode_dir, backup_episode_dir)
                print(f"Successfully moved episode {episode_num:03d} to backup")
            except Exception as e:
                print(f"Error processing episode {episode_num:03d}: {str(e)}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 50)
    print(f"Total episodes processed: {len(missing_videos)}")
    if dry_run:
        print("This was a dry run. No files were actually removed.")
        print("Run with dry_run=False to actually remove the episodes.")
    else:
        print(f"Incomplete episodes have been moved to: {backup_dir}")
        print("You can delete the backup directory manually if needed.")

if __name__ == "__main__":
    # Base directory path - adjust this to match your setup
    base_dir = os.path.expanduser('~/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/rl/act-main-3/experiments/eval/whole_policy/0625-stereo-200_correction/')
    
    # First run with dry_run=True to see what would be removed
    print("\nDry run to show what would be removed:")
    remove_incomplete_episodes(base_dir, dry_run=True)
    
    # Ask for confirmation before actual removal
    response = input("\nDo you want to proceed with removal? (yes/no): ")
    if response.lower() == 'yes':
        print("\nProceeding with actual removal...")
        remove_incomplete_episodes(base_dir, dry_run=False)
    else:
        print("\nAborted. No files were removed.") 