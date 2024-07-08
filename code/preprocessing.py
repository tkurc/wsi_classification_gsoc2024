import os
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image
import streamlit as st
import timm
from transformers import AutoFeatureExtractor


class PreprocessingStrategy:
    def preprocess_and_apply(self, huggingface_dataset, selected_model):
        raise NotImplementedError

class TransformersPreprocessingStrategy(PreprocessingStrategy):
    def preprocess_and_apply(self, huggingface_dataset, selected_model):
        feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model)
        
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

        return train_dataset_dict, val_dataset_dict, test_dataset_dict

class TimmPreprocessingStrategy(PreprocessingStrategy):
    def preprocess_and_apply(self, huggingface_dataset, selected_model):
        model = timm.create_model(selected_model, pretrained=True)
        config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**config)

        def preprocess_images(example):
            image = Image.open(example['file_path']).convert("RGB")
            image = transform(image)
            example['pixel_values'] = image
            return example

        train_dataset_dict = huggingface_dataset['train'].map(preprocess_images)
        train_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])
        
        val_dataset_dict = huggingface_dataset['validation'].map(preprocess_images)
        val_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])
        
        test_dataset_dict = huggingface_dataset['test'].map(preprocess_images)
        test_dataset_dict.set_format(type='torch', columns=['pixel_values', 'label'])

        return train_dataset_dict, val_dataset_dict, test_dataset_dict

class PreprocessingFactory:
    @staticmethod
    def get_preprocessing_strategy(model_type):
        if model_type == 'Transformers':
            return TransformersPreprocessingStrategy()
        elif model_type == 'Timm':
            return TimmPreprocessingStrategy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


