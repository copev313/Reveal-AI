import yaml

def load_config(config_path: str):
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: dict, config_path: str):
    """Saves a dict to a YAML configuration file."""
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)