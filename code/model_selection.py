import yaml
import streamlit as st
from abc import ABC, abstractmethod

# Configuration loading
class ConfigLoader(ABC):
    @abstractmethod
    def load(self, path):
        pass

class YAMLConfigLoader(ConfigLoader):
    def load(self, path):
        with open(path, "r") as file:
            return yaml.safe_load(file)

# Session state
class SessionState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionState, cls).__new__(cls)
            cls._instance.selected_model_timm = None
            cls._instance.selected_model_transformers = None
            cls._instance.training_started = False
        return cls._instance

# Model selection
class ModelSelector(ABC):
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.session_state = SessionState()

    @abstractmethod
    def load_models(self):
        pass

    @abstractmethod
    def select_model(self):
        pass

    def _get_model_option(self):
        return st.radio(
            "Select a model option:",
            ('Select from list', 'Enter custom model path')
        )

    def _display_selected_model(self, selected_model):
        st.write(f"Selected model: {selected_model}")

class TimmModelSelector(ModelSelector):
    """ Timm Model Selector """

    def load_models(self):
        cfg = self.config_loader.load("./config/timm_models.yaml")
        return cfg['Timm_Models']

    def select_model(self):
        st.header("Timm Models")
        timm_models = self.load_models()
        option = self._get_model_option()

        if option == 'Select from list':
            selected_model = st.selectbox(
                "Select any model from the following:",
                ["None"] + list(timm_models.values())
            )
        else:
            selected_model = st.text_input("Enter the custom model path:")

        self.session_state.selected_model_timm = selected_model
        self._display_selected_model(selected_model)
        return selected_model

class TransformersModelSelector(ModelSelector):
    """ Transformers Model Selector """
    def load_models(self):
        cfg = self.config_loader.load("./config/models.yaml")
        return cfg['Vit_Models']

    def select_model(self):
        st.header("Transformer Models")
        transformer_models = self.load_models()
        option = self._get_model_option()

        if option == 'Select from list':
            selected_model = st.selectbox(
                "Select any model from the following:",
                ["None"] + list(transformer_models.values())
            )
        else:
            selected_model = st.text_input("Enter the custom model path:")

        self.session_state.selected_model_transformers = selected_model
        self._display_selected_model(selected_model)
        return selected_model

def model_selection():
    """
    Returns:
    --------
    model: str = Selected model
    model_type: str = Selected model type [Only two type available: Timm, Transformers]
    """

    st.header("Choose a Type of Foundation Models")
    config_loader = YAMLConfigLoader()
    
    model_type = st.radio(
        "Select a model type:",
        ('Timm', 'Transformers')
    )

    if model_type == 'Timm':
        selector = TimmModelSelector(config_loader)
    else:
        selector = TransformersModelSelector(config_loader)
    
    model = selector.select_model()
    return model, model_type
