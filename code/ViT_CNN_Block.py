import torch
import torch.nn as nn
import streamlit as st
from torchsummary import summary
from CNN_Blocks import CustomCNN_01, CustomCNN_02, CustomCNN_03, CustomCNN_04

class ModifiedViT(nn.Module):
    def __init__(self, vit_model):
        super(ModifiedViT, self).__init__()
        self.vit = vit_model
        
    def forward(self, x):
        outputs = self.vit(x)
        return outputs.last_hidden_state

  
# Combine the modified ViT and custom CNN
class ViTWithCustomCNN(nn.Module):
    def __init__(self, vit_model, cnn_model_block, cnn_layers, hidden_channels, model_name = "", num_classes = 2):
        super(ViTWithCustomCNN, self).__init__()
        self.modified_vit = ModifiedViT(vit_model)
        
        if cnn_model_block == "CustomCNN_01":
            self.custom_cnn = CustomCNN_01(vit_model.config.hidden_size, hidden_channels, cnn_layers, num_classes=num_classes)
        elif cnn_model_block == "CustomCNN_02":
            self.custom_cnn = CustomCNN_02(vit_model.config.hidden_size, hidden_channels, cnn_layers, num_classes=num_classes)
        elif cnn_model_block == "CustomCNN_03":
            self.custom_cnn = CustomCNN_03(vit_model.config.hidden_size, hidden_channels, cnn_layers, num_classes=num_classes)
        elif cnn_model_block == "CustomCNN_04":
            self.custom_cnn = CustomCNN_04(vit_model.config.hidden_size, hidden_channels, cnn_layers, num_classes=num_classes)
        else:
            raise ValueError("Please select a valid CNN block")
        
        st.text(summary(self.custom_cnn))
    
    def forward(self, x):
        x = self.modified_vit(x) # Extract features using ViT. X.shape = [batch_size, 197, 768]
        x = x.permute(0, 2, 1).view(x.size(0), x.size(2), 1, -1)  # Reshape for CNN # X.shape = [batch_size, 768, 1, 197]
        x = self.custom_cnn(x)
        return x
