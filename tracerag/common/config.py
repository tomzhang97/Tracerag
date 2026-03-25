"""
Configuration module for TraceRAG.
Handles loading and validating settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, loads default config.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        default_config = Path(__file__).parent.parent / "config" / "defaults.yaml"
        config_path = str(default_config)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return expand_env_vars(config)

def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand ${VAR} references in config values."""
    def expand(value):
        if isinstance(value, str):
            while "${" in value:
                start = value.index("${")
                end = value.index("}", start)
                var_name = value[start+2:end]
                var_value = config.get(var_name, os.environ.get(var_name, ""))
                value = value[:start] + str(var_value) + value[end+1:]
            return value
        elif isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand(item) for item in value]
        else:
            return value
    return expand(config)

def setup_logging(config: Dict[str, Any]):
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get("logging", {})
    level = log_config.get("level", "INFO")
    log_file = log_config.get("file")
    log_format = log_config.get("format",
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), format=log_format, level=level, colorize=True)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, format=log_format, level=level, rotation="100 MB", retention="30 days")

def apply_ablation_overrides(config: Dict[str, Any], ablations: list) -> Dict[str, Any]:
    """
    Apply explicit ablation settings by mutating the config dictionary.
    
    Args:
        config: Base configuration
        ablations: List of ablation keys (e.g., 'no_snap', 'no_vision')
    """
    for override in ablations:
        if override == "no_snap":
            config["retrieval"] = config.get("retrieval", {})
            config["retrieval"]["use_snapper"] = False
        elif override == "no_vision":
            config["visual"] = config.get("visual", {})
            config["visual"]["enabled"] = False
        elif override == "no_vector":
            config["structural"] = config.get("structural", {})
            config["structural"]["enabled"] = False
        else:
            logger.warning(f"Unknown ablation override: {override}")
            
    return config

