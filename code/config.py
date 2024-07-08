import yaml
import json

class ConfigBuilder:
    @staticmethod
    def save_config_yaml(model, optimizer, learning_rate, epochs, batch_size, early_stopping):
        yaml_config = {
            'model': {
                'name': type(model).__name__,
                'hyperparameters': {
                    'optimizer': {
                        'type': 'Adam',
                        'learning_rate': learning_rate,
                        'beta_1': optimizer.defaults['betas'][0],
                        'beta_2': optimizer.defaults['betas'][1]
                    },
                    'loss': 'CrossEntropyLoss',
                    'metrics': [
                        'accuracy'
                    ]
                }
            },
            'training': {
                'batch_size': batch_size,
                'epochs': epochs,
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': early_stopping.patience,
                }
            }
        }

        return yaml_config

    @staticmethod
    def save_config_json(model):
        json_config = {
        "spec_version": "1.0",
        "architecture": model.__class__.__name__ + model.custom_cnn.__class__.__name__,
        "num_classes": 2,
        "class_names": [
        "Other",
        "Tumor"
    ],
        "patch_size_pixels": 350,
        "spacing_um_px": 0.25,
        "transform": [
        {
        "name": "Resize",
        "arguments": {
            "size": 224
        }
        },
        {
        "name": "ToTensor"
        },
        {
        "name": "Normalize",
        "arguments": {
            "mean": [
            0.7238,
            0.5716,
            0.6779
            ],
            "std": [
            0.112,
            0.1459,
            0.1089
            ]
        }
        }
    ]
    }
        return json_config