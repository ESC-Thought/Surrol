import os
import sys
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

cs = ConfigStore.instance()

def print_config_tree(cfg, resolve=False):
    """Print the config tree with resolved values if requested"""
    print("\n" + "=" * 80)
    print("CONFIG TREE")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg, resolve=resolve))
    print("=" * 80 + "\n")

@hydra.main(version_base="1.3", config_path="transic/main/cfg", config_name="residual_config")
def main(cfg: DictConfig) -> None:
    """Main training script for surgical residual policy
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        # Print the raw config
        print_config_tree(cfg, resolve=False)
        
        # Check for required configurations
        if cfg.data_dir == "???" or not os.path.exists(cfg.data_dir):
            print(f"Error: data_dir '{cfg.data_dir}' does not exist or is not specified")
            print("Please provide a valid data directory using: data_dir=/path/to/data")
            return
        
        # Set random seed
        pl.seed_everything(cfg.seed if cfg.seed >= 0 else None)
        
        # Initialize wandb if enabled
        if cfg.get("use_wandb", False) and cfg.get("wandb_project", None):
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.get("wandb_run_name", None),
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        
        # Create checkpoint callbacks
        checkpoint_callbacks = []
        if "checkpoint" in cfg.trainer:
            for ckpt_cfg in cfg.trainer.checkpoint:
                checkpoint_callbacks.append(
                    ModelCheckpoint(**OmegaConf.to_container(ckpt_cfg))
                )
        
        # Create trainer with a copy of the trainer config without checkpoint
        trainer_config = OmegaConf.to_container(cfg.trainer)
        if "checkpoint" in trainer_config:
            del trainer_config["checkpoint"]
        
        # Create trainer
        trainer = hydra.utils.instantiate(
            trainer_config,
            callbacks=checkpoint_callbacks,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        print(f"Creating data module with data_dir: {cfg.data_dir}")
        # Create data module
        try:
            data_module = hydra.utils.instantiate(cfg.data_module)
            print("Data module created successfully")
        except Exception as e:
            print(f"Error creating data module: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("Creating model...")
        # Create model
        try:
            # Check if residual_policy is in the config
            if hasattr(cfg, "residual_policy"):
                residual_policy = hydra.utils.instantiate(cfg.residual_policy)
                print(f"Residual policy instantiated from root config: {cfg.residual_policy._target_}")
            else:
                print("Warning: residual_policy not found in root config.")
                print("Checking if it's defined in the surgical task config...")
                
                # Try to load from surgical task config
                try:
                    from hydra.utils import get_original_cwd
                    import yaml
                    
                    task_path = os.path.join(get_original_cwd(), "transic/main/cfg/residual_policy_task/surgical.yaml")
                    if os.path.exists(task_path):
                        with open(task_path, 'r') as f:
                            task_config = yaml.safe_load(f)
                        
                        if "residual_policy" in task_config:
                            print(f"Found residual_policy in surgical task config: {task_config['residual_policy']['_target_']}")
                            residual_policy = hydra.utils.instantiate(OmegaConf.create(task_config["residual_policy"]))
                        else:
                            raise ValueError("residual_policy not found in surgical task config")
                    else:
                        raise FileNotFoundError(f"Task config not found at {task_path}")
                except Exception as e:
                    print(f"Error loading from task config: {e}")
                    import traceback
                    traceback.print_exc()
                    return
            
            model = hydra.utils.instantiate(
                cfg.module,
                residual_policy=residual_policy,
            )
            print("Model created successfully")
        except Exception as e:
            print(f"Error creating model: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Train
        print("Starting training...")
        trainer.fit(model, data_module)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 