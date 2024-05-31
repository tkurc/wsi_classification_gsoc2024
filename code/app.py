# Importing the libraries
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from transformers import ViTModel
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from stqdm import stqdm
from torch.utils.data import DataLoader

# Title of the app
st.title('Classification of Tumor Infiltrating Lymphocytes in Whole Slide Images of 23 Types of Cancer using Hugging Face')

# Check if 'selected_model' and 'training_started' are in session_state, if not, initialize them
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'training_started' not in st.session_state:
    st.session_state.training_started = False

model_dir_dict = {
    "vit-base-patch16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-base-patch32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-large-patch16-224-in21k": "google/vit-large-patch16-224-in21k",
    "dinov2-base": "facebook/dinov2-base",
    "dino-vitb8": "facebook/dino-vitb8",
    "dino-vits8": "facebook/dino-vits8",
    "deit-base-distilled-patch16-224": "facebook/deit-base-distilled-patch16-224",
    "vit-dino16": "facebook/dino-vits16"
}

# Create the selectbox for model selection
selected_model = st.selectbox(
    "Select any model from the following:",
    ["None"] + list(model_dir_dict.keys())
)

# Add a button to start training
if selected_model != "None":
    if st.button("Start Training"):
        # Update the session_state with the selected model
        st.session_state.selected_model = selected_model
        st.session_state.training_started = True

# Check if training should start
if st.session_state.training_started:
    st.write(f"Selected model: {st.session_state.selected_model}")

    # Load the model
    model = ViTModel.from_pretrained(model_dir_dict[st.session_state.selected_model])

    # Preparing the Dataset
    from create_dataset import ThymDataset
    data_class = ThymDataset()
    df = data_class.load_data()

    # Function to plot images with labels
    def plot_images_with_labels(df, num_images=5):
        sample = df.sample(n=num_images).reset_index(drop=True)
        fig, axes = plt.subplots(1, num_images, figsize=(35, 25))
        for i in range(num_images):
            file_path = sample.loc[i, 'file_path']
            label = sample.loc[i, 'label']
            image = Image.open(file_path)
            ax = axes[i]
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'Label: {label}', fontsize=40)
        st.pyplot(fig)

    st.title("Image Label Viewer")
    if 'df' in locals():
        num_images = st.slider("Select the number of images to display:", min_value=1, max_value=20, value=5)
        plot_images_with_labels(df, num_images=num_images)
    else:
        st.write("DataFrame 'df' is not defined. Please load the data.")

    # Convert the data to Hugging Face dataset format
    huggingface_dataset = data_class.convert_to_huggingface_dataset_format()
    from transformers import AutoFeatureExtractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir_dict[st.session_state.selected_model])

    def preprocess_images(example):
        image = Image.open(example['file_path']).convert("RGB")
        encoding = feature_extractor(images=image, return_tensors="pt")
        example['pixel_values'] = encoding['pixel_values'][0]
        return example

    train_dataset_dict = huggingface_dataset['train'].map(preprocess_images)
    train_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])
    val_dataset_dict = huggingface_dataset['validation'].map(preprocess_images)
    val_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])
    test_dataset_dict = huggingface_dataset['test'].map(preprocess_images)
    test_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])

    # Define the model
    class ViTWithCNNHead(nn.Module):
        def __init__(self, vit_model):
            super(ViTWithCNNHead, self).__init__()
            self.vit = vit_model
            self.cnn_head = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 2)
            )

        def forward(self, pixel_values):
            with torch.no_grad():
                vit_outputs = self.vit(pixel_values=pixel_values)
            cls_output = vit_outputs.last_hidden_state[:, 0]
            cls_output = cls_output.unsqueeze(1).unsqueeze(2)
            cnn_output = self.cnn_head(cls_output)
            return cnn_output

    # Load the ViT model and freeze its parameters
    vit_model = ViTModel.from_pretrained(model_dir_dict[st.session_state.selected_model])
    for param in vit_model.parameters():
        param.requires_grad = False

    # Instantiate the custom model
    model = ViTWithCNNHead(vit_model)

    # Training setup
    epochs = 10
    batch_size = 16
    learning_rate = 1e-3

    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'pixel_values': pixel_values, 'label': labels}

    train_loader = DataLoader(train_dataset_dict, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(val_dataset_dict, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset_dict, batch_size=batch_size, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    st.title("Model Training Progress")
    epoch_text = st.empty()
    validation_text = st.empty()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

        avg_validation_loss = validation_loss / len(validation_loader)
        accuracy = 100 * correct / total
        validation_text.text(f"Validation Loss: {avg_validation_loss:.4f}, Accuracy: {accuracy:.2f}%")

    st.write("Training completed")
else:
    st.write("Please select a model from the list above and click 'Start Training'.")
