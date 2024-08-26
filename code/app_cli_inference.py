# Importing the libraries
import yaml
import json
import subprocess
import time
import streamlit as st
from stqdm import stqdm
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# From the local imports
from dataset import load_dataset
from preprocessing import PreprocessingFactory
from model_selection import model_selection
from model_setup import ModelSetup, ModelSetupCli
from training import training, TrainingConfigFactory
from earlystopping import EarlyStopping
from wsi_infer import run_wsinfer

import click
from click.testing import CliRunner
from wsinfer.cli.infer import run

@click.command()
@click.option(
    "-i",
    "--wsi_dir",
    required=True,
    prompt='Select Your Slides directory Path', 
    default="./slides/",
    help="Directory containing whole slide images. This directory can *only* contain"
    " whole slide images.",
)

@click.option(
    "-o",
    "--results_dir",
    prompt='Select Your Results directory Path', 
    required=True,
    default="./results/",
    help="Directory to store results. If directory exists, will skip"
    " whole slides for which outputs exist.",
)

@click.option(
    "-c",
    "--config_path",
    prompt='Select Your Config file Path', 
    required=True,
    help=(
        "Path to configuration for the trained model."
        "You can find the configuration file under metadata folder. This file should be json format."
    ),
)
@click.option(
    "-p",
    "--model_path",
    prompt='Select Your Saved model Path',
    required=True,
    help=(
        "Path to the trained model. Use only when --config is passed. Mutually "
        "exclusive with --model. Find the model under the metadata folder name model.pt." 
    ),
)


def inference(wsi_dir, results_dir, config_path, model_path):   
    # Inference
    # print("Starting Inference")
    # run_wsinfer(wsi_dir, results_dir, config, model_path)
    # print("Inference Completed!")
    runner = CliRunner()

    # Run the command
    """
    CliRunner object, which simulates running a command from the command line.
    We use runner.invoke() to run the command, passing the run function and a list of arguments.
    """
    result = runner.invoke(run, [
        '--wsi-dir', wsi_dir,
        '--results-dir', results_dir,
        '--config', config_path,
        '--model-path', model_path,

    ])

    # Check the result
    if result.exit_code == 0:
        print("Command executed successfully")
    
    else:
        print(f"Command failed with exit code {result.exit_code}")
        print(result.output)

if __name__ == '__main__':
    inference()


# python app_cli_inference.py -i ./slides/brca/ -o ./results/brca/ -c ./metadata/10h_29m_30s_17082024/metadata.json -p ./metadata/10h_29m_30s_17082024/model.pt