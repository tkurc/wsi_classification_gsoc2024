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
from wsi_infer import wsi_infer, run_wsinfer

import click


@click.command()
@click.option('-mt', '--model_type', 
            prompt='Select Your model type', 
            default= "Timm",
            show_choices=["Transformers", "Timm"], 
            help='Select any model type.')

@click.option('-m', '--model', 
            prompt='Choose Your model', 
            default= "None",
            help='Select any model from the following.')


@click.option('-cnnb', '--cnn_block', 
            prompt='Choose any CNN block', 
            default= "CustomCNN_01",
            show_choices=["CustomCNN_01", "CustomCNN_02", "CustomCNN_03", "CustomCNN_04"],
            help='Select any CNN Block from the following.')


@click.option('-cnnl', '--cnn_layers', 
            prompt='Choose the number of CNN layers', 
            default= 2,
            show_choices=[2,3,4,5,6],
            help='Select number of CNN layers.')


@click.option('-cnnf', '--cnn_features', 
            prompt='Choose the number of CNN feature', 
            default= 32,
            show_choices=[32,64,128, 256,512,1024],
            help='Select number of CNN feature.')


@click.option('-e', '--epochs', 
            prompt='Choose the number of epochs', 
            default= 1,
            show_choices=[1,2,5,10, 20, 30, 40, 50, 100],
            help='Select number of epochs.')

@click.option('-bs', '--batch_size', 
            prompt='Choose the batch size', 
            default= 16,
            show_choices=[16, 32, 64, 128, 256, 512],
            help='Select Batch Size.')

@click.option('-es', '--early_stopping', 
            prompt='Choose the early stopping', 
            default= 1,
            show_choices=[1, 2, 3, 4, 5, 6],
            help='Select Early Stopping.')



def main(model_type, model, cnn_block, cnn_layers, cnn_features, epochs, batch_size, early_stopping):
    
    # We already know the model_type and model
    if model != "None":
        # Load the dataset
        dataset = load_dataset()

        # Preprocessing the dataset
        preprocessing_strategy = PreprocessingFactory.get_preprocessing_strategy(model_type)
        train_dataset, val_dataset, test_dataset = preprocessing_strategy.preprocess_and_apply(dataset, model)
        
        # Training
        try:
            model_setup = ModelSetupCli(model, model_type, cnn_block, cnn_layers, cnn_features)
            training_model = model_setup.setup()
            early_stopping_value = EarlyStopping(patience=early_stopping, verbose=True)
            training(training_model, train_dataset, val_dataset, test_dataset, epochs, batch_size, early_stopping_value)
            print('Training Completed!')
        except Exception as e:
            print(f"An error occurred: {e}")
            print('Training Failed!')
    else:
        print("Please select a model")

    





# google/vit-base-patch16-224-in21k
# Transformers
# python app_cli.py -mt Transformers -m google/vit-base-patch16-224-in21k -cnnb CustomCNN_01 -cnnl 2 -cnnf 32 -e 5 -bs 16 -es 3 -i ./slides/brca/ -o ./results/brca/ -c ./metadata/10h_29m_30s_17082024/metadata.json -p ./metadata/10h_29m_30s_17082024/model.pt