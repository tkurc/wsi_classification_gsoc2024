import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from stqdm import stqdm
from utils import create_timestamped_dir, save_config_to_yaml,  save_metadata_to_json
from earlystopping import EarlyStopping
import yaml
import json
from config import ConfigBuilder

class TrainingConfig:
    def __init__(self, epochs, batch_size, early_stopping):
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping

    def __str__(self):
        return f"TrainingConfig(epochs={self.epochs}, batch_size={self.batch_size}, early_stopping={self.early_stopping})"

class TrainingConfigFactory:
    @staticmethod
    def create_config():
        if "epochs" not in st.session_state:
            st.session_state.epochs = 0
        if "batch_size" not in st.session_state:
            st.session_state.batch_size = 0
        if "early_stopping_patience" not in st.session_state:
            st.session_state.early_stopping_patience = 0

        epochs = st.selectbox("Enter the number of epochs:", [1, 5, 10, 20, 50, 100])
        st.session_state.epochs = epochs
        st.write("You have selected number of epochs:", epochs)

        batch_size = st.selectbox("Enter the batch size:", [16, 32, 64, 128, 256])
        st.session_state.batch_size = batch_size
        st.write("You have selected batch size:", batch_size)

        early_stopping_patience = st.selectbox("Select the number of epochs to wait before stopping the training:", [2, 5, 10, 20, 30, 40, 50])
        st.session_state.early_stopping_patience = early_stopping_patience
        st.write("You have selected early stopping patience:", early_stopping_patience)

        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

        return TrainingConfig(epochs, batch_size, early_stopping)


   

def training(model, train_dataset_dict, val_dataset_dict, test_dataset_dict, epochs, batch_size, early_stopping):

    learning_rate = 1e-3
    print("Model Type:", type(model))

    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'pixel_values': pixel_values, 'label': labels}

    train_loader = DataLoader(train_dataset_dict, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(val_dataset_dict, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset_dict, batch_size=batch_size, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Try different optimizers such as SGD, AdamW, etc.
    criterion = nn.CrossEntropyLoss()

    st.title("Model Training Progress")
    epoch_text = st.empty()
    validation_text = st.empty()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Save the configuration
    save_dir = create_timestamped_dir("./metadata")
    
    config_yml_data = ConfigBuilder.save_config_yaml(model, optimizer, learning_rate, epochs, batch_size, early_stopping) # Save the configuration in YAML format
    save_config_to_yaml(config_yml_data, save_dir)

    config_json_data = ConfigBuilder.save_config_json(model) # Save the configuration in JSON format
    save_metadata_to_json(config_json_data, save_dir)
    

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_len = len(train_loader)

        for batch in stqdm(train_loader, backend=False, frontend=True, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / train_loader_len
        epoch_text.text(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in validation_loader:
                inputs = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)  
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            early_stopping(validation_loss, model, save_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        avg_validation_loss = validation_loss / len(validation_loader)
        accuracy = 100 * correct / total
        validation_text.text(f"Validation Loss: {avg_validation_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Test the model
    test_loader_len = len(test_loader)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)  
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    st.write(f"Test Accuracy: {test_accuracy:.2f}%")
