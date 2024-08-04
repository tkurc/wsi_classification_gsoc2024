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
from model_setup import ModelSetup
from training import training, TrainingConfigFactory
from wsi_infer import wsi_infer


def main():
    st.title('Classification of Tumor Infiltrating Lymphocytes in Whole Slide Images of 23 Types of Cancer using Hugging Face')

    # Initialize session state
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False

    # Training section
    if not st.session_state.training_completed:
        
        selected_model, selected_model_type = model_selection()

        if selected_model != "None":
            dataset = load_dataset()

            # Model Setup
            model_setup = ModelSetup(selected_model, selected_model_type)
            model = model_setup.setup()

            # Training Config
            training_config = TrainingConfigFactory.create_config()

            if st.button("Start Training"):
                st.subheader("Training")
                # Update the session_state with the selected model
                st.session_state.training_started = True
                st.session_state.training_completed = False

                # Process the dataset
                with st.spinner('Preprocessing images!! Wait for it...'):
                    preprocessing_strategy = PreprocessingFactory.get_preprocessing_strategy(selected_model_type)
                    train_dataset, val_dataset, test_dataset = preprocessing_strategy.preprocess_and_apply(dataset, selected_model)
                    total_time = len(train_dataset) + len(val_dataset) + len(test_dataset)
                    map_time = total_time // 220
                    time.sleep(map_time)
                st.success('Preprocessing Done!')

                # Train the selected model
                try:
                    training(model, train_dataset, val_dataset, test_dataset, training_config.epochs, training_config.batch_size, training_config.early_stopping)
                    st.session_state.training_completed = True
                    st.success('Training Completed!')
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.training_completed = False
        else:
            st.write("Please select a model")
        
    time.sleep(30)
    # Inference section
    if st.session_state.training_completed:
        st.subheader("Inference")
        st.write("Training is complete. You can now proceed to inference.")
        wsi_infer()

if __name__ == "__main__":
    main()