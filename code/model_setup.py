import streamlit as st
import yaml
import timm
from transformers import AutoModel
from abc import ABC, abstractmethod
from ViT_CNN_Block import ViTWithCustomCNN
from TimmWithCNNBlock import TimmWithCustomCNN

class ModelSetupStrategy(ABC):
    @abstractmethod
    def create_base_model(self, model_name):
        pass

    @abstractmethod
    def create_custom_model(self, base_model, cnn_model_block, cnn_layers, hidden_channels):
        pass

class ViTModelSetupStrategy(ModelSetupStrategy):
    def create_base_model(self, model_name):
        vit_model = AutoModel.from_pretrained(model_name)
        for param in vit_model.parameters():
            param.requires_grad = False  # Freeze encoder weights
        return vit_model

    def create_custom_model(self, base_model, cnn_model_block, cnn_layers, hidden_channels):
        return ViTWithCustomCNN(base_model, cnn_model_block, cnn_layers, hidden_channels)

class TimmModelSetupStrategy(ModelSetupStrategy):
    def create_base_model(self, model_name):
        timm_model = timm.create_model(model_name, pretrained=True)
        for param in timm_model.parameters():
            param.requires_grad = False  # Freeze encoder weights
        return timm_model

    def create_custom_model(self, base_model, cnn_model_block, cnn_layers, hidden_channels):
        return TimmWithCustomCNN(base_model, cnn_model_block, cnn_layers, hidden_channels)

class ModelSetupUI:
    """ 
    Model setup: Custom Input from the User
    
    Options:
    --------
        CNN block
        Number of CNN layers
        Number of hidden channels

    """
    def __init__(self):
        if 'cnn_block' not in st.session_state:
            st.session_state.cnn_block = "CustomCNN_01"
        if 'cnn_layers' not in st.session_state:
            st.session_state.cnn_layers = 2
        if 'cnn_features' not in st.session_state:
            st.session_state.cnn_features = 64

    def get_cnn_parameters(self):
        st.header("Custom CNN Block")

        st.session_state.cnn_block = st.radio(
            "Select the CNN block:",
            ["CustomCNN_01", "CustomCNN_02", "CustomCNN_03", "CustomCNN_04"],
            index=["CustomCNN_01", "CustomCNN_02", "CustomCNN_03", "CustomCNN_04"].index(st.session_state.cnn_block)
        )
        st.write(f"Selected CNN block: {st.session_state.cnn_block}")

        st.session_state.cnn_layers = st.selectbox(
            "Enter the number of CNN layers: ",
            [2, 3, 4, 5, 6],
            index=[2, 3, 4, 5, 6].index(st.session_state.cnn_layers)
        )
        st.write("Selected number of layers is : ", st.session_state.cnn_layers)

        st.session_state.cnn_features = st.selectbox(
            "Enter the number of features: ",
            [64, 128, 256, 512, 768, 1024, 2048],
            index=[64, 128, 256, 512, 768, 1024, 2048].index(st.session_state.cnn_features)
        )
        st.write("Selected hidden dimension is: ", st.session_state.cnn_features)

        return st.session_state.cnn_block, st.session_state.cnn_layers, st.session_state.cnn_features


class ModelSetup:
    """ 
    Parameters:
    ------
        model: str =  Model
        model_type: str =  Transformers or Timm

    Returns:
    -------
        model: Model = Custom model with ViT or Timm backbone and custom CNN block
    """
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type
        self.ui = ModelSetupUI()
        self.strategy = self._get_strategy()

    def _get_strategy(self):
        if self.model_type == 'Transformers':
            return ViTModelSetupStrategy()
        elif self.model_type == 'Timm':
            return TimmModelSetupStrategy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def setup(self):
        base_model = self.strategy.create_base_model(self.model_name)
        cnn_model_block, cnn_layers, hidden_channels = self.ui.get_cnn_parameters()
        model = self.strategy.create_custom_model(base_model, cnn_model_block, cnn_layers, hidden_channels)

        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.write('Number of trainable parameters:', '{0:.3f} Million'.format(trainable_parameters/1000000))

        return model
