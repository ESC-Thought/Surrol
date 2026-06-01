import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

"""
This is a minimal test script to isolate Hydra configuration loading issues.
It uses a simple config structure without any complex interpolations.
"""

@hydra.main(version_base="1.3", config_path=".", config_name="test_config")
def main(cfg: DictConfig) -> None:
    print("Configuration loaded successfully!")
    print(OmegaConf.to_yaml(cfg))
    
    # Access some values to test interpolation
    print(f"gpus: {cfg.gpus}")
    print(f"trainer.devices: {cfg.trainer.devices}")

if __name__ == "__main__":
    # First, create a test config file
    with open("test_config.yaml", "w") as f:
        f.write("""
# Simple test configuration
gpus: 1

trainer:
  devices: ${gpus}
  accelerator: "gpu"
        """)
    
    print("Created test_config.yaml")
    
    # Run the main function
    main() 