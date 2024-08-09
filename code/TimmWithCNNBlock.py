import torch
import torch.nn as nn
import streamlit as st
from torchsummary import summary

from CNN_Blocks import CustomCNN_01, CustomCNN_02, CustomCNN_03, CustomCNN_04

class ModifiedTimm(nn.Module):
    def __init__(self, timm_model):
        super(ModifiedTimm, self).__init__()
        self.timm = timm_model
        self.fc = self.timm.get_classifier() # For other timm models
        
        """self.uni = self.timm.blocks[23].mlp.fc2.out_features # unipath # 1024
        self.gigapath = timm_model.blocks[39].mlp.fc2.out_features #gigapath # 1536
        self.virchow =  timm_model.blocks[31].mlp.fc2.out_features # virchow #out_features=1280"""
        if timm_model in ["hf-hub:paige-ai/Virchow", "hf_hub:prov-gigapath/prov-gigapath", "hf-hub:MahmoodLab/uni"]:
        	self.blocks_fc = timm_model.blocks[-1].mlp.fc2.out_features # For Virchow model, unipath, gigapath 
        

    def forward(self, x):
        outputs = self.timm.forward_features(x)
        return outputs

class TimmWithCustomCNN(nn.Module):
    def __init__(self, timm_model, cnn_model_block, cnn_layers, hidden_channels, model_name = "", num_classes=2):
        super(TimmWithCustomCNN, self).__init__()

        self.modified_timm = ModifiedTimm(timm_model)
        self.model_name = model_name

        # For Virchow, UNI, gigapath
        if model_name in ["hf-hub:paige-ai/Virchow", "hf_hub:prov-gigapath/prov-gigapath", "hf-hub:MahmoodLab/uni"]:
            if cnn_model_block == "CustomCNN_01":
                self.custom_cnn = CustomCNN_01(self.modified_timm.blocks_fc, hidden_channels, cnn_layers, num_classes=num_classes)
            elif cnn_model_block == "CustomCNN_02":
                self.custom_cnn = CustomCNN_02(self.modified_timm.blocks_fc, hidden_channels, cnn_layers, num_classes=num_classes)
            elif cnn_model_block == "CustomCNN_03":
                self.custom_cnn = CustomCNN_03(self.modified_timm.blocks_fc, hidden_channels, cnn_layers, num_classes=num_classes)
            elif cnn_model_block == "CustomCNN_04":
                self.custom_cnn = CustomCNN_04(self.modified_timm.blocks_fc, hidden_channels, cnn_layers, num_classes=num_classes)
            else:
                raise ValueError("Invalid CNN model block")
        
        # For other timm models
        else:
            if cnn_model_block == "CustomCNN_01":
                self.custom_cnn = CustomCNN_01(self.modified_timm.fc.in_features, hidden_channels, cnn_layers, num_classes=num_classes)
            elif cnn_model_block == "CustomCNN_02":
                self.custom_cnn = CustomCNN_02(self.modified_timm.fc.in_features, hidden_channels, cnn_layers, num_classes=num_classes)
            elif cnn_model_block == "CustomCNN_03":
                self.custom_cnn = CustomCNN_03(self.modified_timm.fc.in_features, hidden_channels, cnn_layers, num_classes=num_classes)
            elif cnn_model_block == "CustomCNN_04":
                self.custom_cnn = CustomCNN_04(self.modified_timm.fc.in_features, hidden_channels, cnn_layers, num_classes=num_classes) # For other timm models
            else:
                raise ValueError("Invalid CNN model block")
        
    
    def forward(self, x):
        if self.model_name in ["hf-hub:paige-ai/Virchow", "hf_hub:prov-gigapath/prov-gigapath", "hf-hub:MahmoodLab/uni"]:    
            x = self.modified_timm(x)
            x = x.unsqueeze(1) # Adding for UNI and gigapath remove if other models are used
            x = x.permute(0, 3, 1, 2) # adding for UNI and gigapath remove if other models are used
            x = self.custom_cnn(x)
            return x
        else:
            x = self.modified_timm(x)
            x = self.custom_cnn(x)
            return x


