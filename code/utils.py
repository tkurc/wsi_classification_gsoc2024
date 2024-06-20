# Function to create a directory based on current timestamp
import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

def create_timestamped_dir(base_path):
    now = datetime.now()
    folder_name = now.strftime("%Hh_%Mm_%Ss_%d%m%Y")
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# Example function to generate and save configuration to YAML
def save_config_to_yaml(config, save_dir):
    config_file = os.path.join(save_dir, 'config.yml')
    with open(config_file, 'w') as file:
        yaml.dump(config, file, indent=4)
    print(f"Saved configuration to {config_file}")

# Example function to generate and save metadata to JSON
def save_metadata_to_json(metadata, save_dir):
    metadata_file = os.path.join(save_dir, 'metadata.json')
    with open(metadata_file, 'w') as file:
        json.dump(metadata, file, indent=4)
    print(f"Saved metadata to {metadata_file}")

# Example function to save PyTorch model
def save_model(model, save_dir):
    model_file = os.path.join(save_dir, 'model.pt')
    model_jit = torch.jit.trace(model, torch.randn(16, 3, 224, 224).cuda())
    torch.jit.save(model_jit, model_file)
    print(f"Saved PyTorch model to {model_file}")
