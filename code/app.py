# Importing the libraries
import time
import streamlit as st
from stqdm import stqdm
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import yaml
import json
import subprocess

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# From the local imports
from dataset import ThymDataset
from ViT_CNN_Block import ViTWithCustomCNN
from training import training, training_config
from wsi_infer import wsi_infer

from CNN_Blocks import CustomCNN_01, CustomCNN_02, CustomCNN_03, CustomCNN_04

# Hugging Face imports
from transformers import ViTModel, AutoFeatureExtractor

# Load the Data
def load_dataset():
    data_class = ThymDataset()
    df = data_class.load_data()
    
    # Convert the data to Hugging Face dataset format
    huggingface_dataset = data_class.convert_to_huggingface_dataset_format()
    return huggingface_dataset

# Select the model
def load_config():
    with open("./config/models.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    return cfg

def model_selection(cfg):
    Vit_Models = cfg['Vit_Models']
    
    st.header("Foundation Models")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False 

    # User selects model from the list or chooses to input a custom model
    option = st.radio(
        "Select a model option:",
        ('Select from list', 'Enter custom model path')
    )

    if option == 'Select from list':
        selected_model = st.selectbox(
            "Select any model from the following:",
            ["None"] + list(Vit_Models.values())
        )
    else:
        selected_model = st.text_input("Enter the custom model path:")

    # Store the selected model in session state
    st.session_state.selected_model = selected_model

    # Display the selected model
    st.write(f"Selected model: {selected_model}")
    return selected_model


# Apply preocessing on the images
def preprocess_and_apply(huggingface_dataset, selected_model):
    feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model)
    
    def preprocess_images(example):
        image = Image.open(example['file_path']).convert("RGB")
        encoding = feature_extractor(images=image, return_tensors="pt")
        example['pixel_values'] = encoding['pixel_values'][0]
        return example

    # Apply the preprocess_images function to the train, validation, and test sets
    train_dataset_dict = huggingface_dataset['train'].map(preprocess_images)
    train_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])
    
    val_dataset_dict = huggingface_dataset['validation'].map(preprocess_images)
    val_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])
    
    test_dataset_dict = huggingface_dataset['test'].map(preprocess_images)
    test_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])

    return train_dataset_dict, val_dataset_dict, test_dataset_dict



# Define the Model
def model_setup(model_dir):
    vit_model = ViTModel.from_pretrained(model_dir)
    # Freeze ViT encoder layers
    for param in vit_model.parameters():
        param.requires_grad = False  # Freeze encoder weights
    
    st.header("Custom CNN Block")

    if "cnn_layers" not in st.session_state:
        st.session_state.cnn_layers = 0
    
    if "cnn_features" not in st.session_state:
        st.session_state.cnn_features = 0
    
    if "cnn_block" not in st.session_state:
        st.session_state.cnn_block = None

    # Select the CNN block
    cnn_block = st.radio(
        "Select the CNN block:",
        ["CustomCNN_01", "CustomCNN_02", "CustomCNN_03", "CustomCNN_04"]
    )
    # Store the selected CNN block in session state
    st.session_state.cnn_block = cnn_block
    st.write(f"Selected CNN block: {cnn_block}")
    
    # Select the number of CNN layers
    cnn_layers = st.selectbox(
        "Enter the number of CNN layers: ",
        [2, 3, 4, 5 , 6],
    )
    # Store the layers in session state
    st.session_state.cnn_layers = cnn_layers
    st.write("Selected number of layers is : ", cnn_layers)

    # Select the number of features
    cnn_features = st.selectbox(
        "Enter the number of features: ",
        [64, 128, 256, 512, 768, 1024, 2048],
    )
    # Store the features in session state
    st.session_state.cnn_features = cnn_features
    st.write("Selected hidden dimension is: ", cnn_features)
    

    model = ViTWithCustomCNN(vit_model, cnn_block, cnn_layers, cnn_features)
    
    # Number of trainable parameters
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    st.write('Number of trainable parameters:', '{0:.3f} Million'.format(trainable_parameters/1000000 ))
    
    return model

def main():
    st.title('Classification of Tumor Infiltrating Lymphocytes in Whole Slide Images of 23 Types of Cancer using Hugging Face')

    # Initialize session state
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False

    # Training section
    if not st.session_state.training_completed:
        st.subheader("Training")
        cfg = load_config()
        selected_model = model_selection(cfg)
        
        if selected_model != "None":
            my_dataset = load_dataset()
            model = model_setup(selected_model)
            epochs, batch_size, early_stopping = training_config()

            if st.button("Start Training"):
                # Update the session_state with the selected model
                st.session_state.training_started = True
                st.session_state.training_completed = False

                # Process the dataset
                with st.spinner('Preprocessing images!! Wait for it...'):
                    train_dataset_dict, val_dataset_dict, test_dataset_dict = preprocess_and_apply(my_dataset, selected_model)
                    total_time = len(train_dataset_dict) + len(val_dataset_dict) + len(test_dataset_dict)
                    map_time = total_time // 240
                    time.sleep(map_time)
                st.success('Preprocessing Done!')

                # Load the selected model
                try:
                    training(model, train_dataset_dict, val_dataset_dict, test_dataset_dict, epochs, batch_size, early_stopping)
                    st.session_state.training_completed = True
                    st.success('Training Completed!')
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.training_completed = False
        else:
            st.write("Please select a model")

    # Inference section
    if st.session_state.training_completed:
        st.subheader("Inference")
        st.write("Training is complete. You can now proceed to inference.")
        wsi_infer()

if __name__ == "__main__":
    main()