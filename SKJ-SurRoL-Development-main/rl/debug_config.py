import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def check_config_path(path, base_dir=None):
    """Check if a config path exists"""
    if base_dir:
        full_path = os.path.join(base_dir, path)
    else:
        full_path = path
        
    exists = os.path.exists(full_path)
    print(f"Checking path: {full_path} - {'EXISTS' if exists else 'NOT FOUND'}")
    return exists, full_path

@hydra.main(version_base="1.3", config_path="transic/main/cfg", config_name="residual_config")
def main(cfg: DictConfig) -> None:
    """Debug script to test configuration loading
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        print_section("CONFIGURATION LOADED SUCCESSFULLY")
        print(OmegaConf.to_yaml(cfg))
        
        # Print key configuration values
        print_section("KEY CONFIGURATION VALUES")
        print(f"- data_dir: {cfg.data_dir}")
        print(f"- gpus: {cfg.gpus}")
        print(f"- seed: {cfg.seed}")
        print(f"- use_wandb: {cfg.use_wandb}")
        
        # Check if residual_policy is properly configured
        print_section("RESIDUAL POLICY CONFIGURATION")
        if hasattr(cfg, "residual_policy"):
            print(f"- Found in root config")
            print(f"- Target: {cfg.residual_policy._target_}")
            print(f"- Action dim: {cfg.residual_policy.action_dim}")
        else:
            print("- Not found in root config")
            
            # Check if it's in the module config
            if hasattr(cfg.module, "residual_policy"):
                print("- Found in module config")
                print(f"- Target: {cfg.module.residual_policy._target_}")
            else:
                print("- Not found in module config")
            
            # Check if it's in the task config
            print("\nChecking task config files:")
            from hydra.utils import get_original_cwd
            cwd = get_original_cwd()
            
            # Check surgical task config
            task_path = "transic/main/cfg/residual_policy_task/surgical.yaml"
            exists, full_path = check_config_path(task_path, cwd)
            
            if exists:
                with open(full_path, 'r') as f:
                    task_config = yaml.safe_load(f)
                
                if "residual_policy" in task_config:
                    print(f"- Found in surgical task config")
                    print(f"- Target: {task_config['residual_policy']['_target_']}")
                    print(f"- Action dim: {task_config['residual_policy'].get('action_dim', 'Not specified')}")
                else:
                    print("- Not found in surgical task config")
        
        # Check architecture config
        print_section("ARCHITECTURE CONFIGURATION")
        arch_name = cfg.arch_name
        print(f"- Architecture name: {arch_name}")
        
        arch_path = f"transic/main/cfg/residual_policy_arch/{arch_name}.yaml"
        exists, full_path = check_config_path(arch_path, cwd)
        
        if exists:
            with open(full_path, 'r') as f:
                arch_config = yaml.safe_load(f)
            
            if "residual_policy" in arch_config:
                print(f"- Found residual_policy in architecture config")
                print(f"- Target: {arch_config['residual_policy']['_target_']}")
            else:
                print("- residual_policy not found in architecture config")
        
        # Check module configuration
        print_section("MODULE CONFIGURATION")
        print(f"- Target: {cfg.module._target_}")
        print(f"- Learning rate: {cfg.module.lr}")
        
        # Check trainer configuration
        print_section("TRAINER CONFIGURATION")
        print(f"- Target: {cfg.trainer._target_}")
        print(f"- Devices: {cfg.trainer.devices}")
        print(f"- Checkpoint configs: {len(cfg.trainer.checkpoint)}")
        
        # Check defaults structure
        print_section("DEFAULTS STRUCTURE")
        if hasattr(cfg, "defaults"):
            print(f"- Defaults: {cfg.defaults}")
            for i, default in enumerate(cfg.defaults):
                if isinstance(default, dict):
                    for k, v in default.items():
                        print(f"  - {i}: {k}: {v}")
                else:
                    print(f"  - {i}: {default}")
        else:
            print("- No defaults found in config")
            
    except Exception as e:
        print(f"Error in debug script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 