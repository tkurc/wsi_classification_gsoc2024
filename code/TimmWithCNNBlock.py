import torch
import torch.nn as nn
import streamlit as st
from torchsummary import summary
import timm
from CNN_Blocks import CustomCNN_01, CustomCNN_02, CustomCNN_03, CustomCNN_04

class ModifiedTimm(nn.Module):
    def __init__(self, timm_model):
        super(ModifiedTimm, self).__init__()
        self.timm = timm_model
        self.fc = self.timm.get_classifier()

    def forward(self, x):
        outputs = self.timm.forward_features(x)
        return outputs

class TimmWithCustomCNN(nn.Module):
    def __init__(self, timm_model, cnn_model_block, cnn_layers, hidden_channels, num_classes=2):
        super(TimmWithCustomCNN, self).__init__()
        self.modified_timm = ModifiedTimm(timm_model)
        if cnn_model_block == "CustomCNN_01":
            self.custom_cnn = CustomCNN_01(self.modified_timm.fc.in_features, hidden_channels, cnn_layers, num_classes=num_classes)
        elif cnn_model_block == "CustomCNN_02":
            self.custom_cnn = CustomCNN_02(self.modified_timm.fc.in_features, hidden_channels, cnn_layers, num_classes=num_classes)
        elif cnn_model_block == "CustomCNN_03":
            self.custom_cnn = CustomCNN_01(self.modified_timm.fc.in_features, hidden_channels, cnn_layers, num_classes=num_classes)
        elif cnn_model_block == "CustomCNN_04":
            self.custom_cnn = CustomCNN_02(self.modified_timm.fc.in_features, hidden_channels, cnn_layers, num_classes=num_classes)

        else:
            raise ValueError("Invalid CNN model block")
    
    def forward(self, x):
        x = self.modified_timm(x) # Output shape: [batch_size, 7, 7, 2048]
        x = self.custom_cnn(x)
        return x