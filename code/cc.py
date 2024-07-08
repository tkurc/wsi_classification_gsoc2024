# import torch
# import torch.nn as nn

# class CustomCNN(nn.Module):
#     def __init__(self, num_layers, input_channels, hidden, num_classes=2):
#         super(CustomCNN, self).__init__()
#         layers = []
#         layers.append(nn.Conv2d(input_channels, hidden, kernel_size=3, stride=1, padding=1))
#         for i in range(num_layers):
#             layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1))
#             layers.append(nn.ReLU())
#         self.cnn = nn.Sequential(*layers)
#         print("CNN:",self.cnn)
#         self.classifier = nn.Linear(hidden, num_classes)
    
#     def forward(self, x):
#         print(x.shape)
#         # Ensure the input has 4 dimensions [batch_size, channels, height, width]
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)  # Adding a channel dimension
#         x = self.cnn(x)
#         x = x.mean(dim=[2, 3])  # Global average pooling
#         x = self.classifier(x)
#         return x

# # Instantiate the model
# c = CustomCNN(num_layers=2, input_channels=3, hidden = 64, num_classes=2)

# # Dummy input for testing (batch size of 3, 3 color channels, 224x224 image size)
# dummy_input = torch.randn(16, 3, 224, 224)

# # Forward pass through the model
# output = c(dummy_input)
# print(output.shape)  # Should output torch.Size([3, 2])





#     # def forward(self, x):
#     #     x = x.permute(0, 2, 1).view(x.size(0), x.size(2), 1, -1)  # Reshape for CNN
#     #     x = self.cnn(x)
#     #     x = x.mean(dim=[2, 3])  # Global average pooling
#     #     x = self.classifier(x)
#     #     return x



# import torch
# import torch.nn as nn
# import streamlit as st
# from torchsummary import summary
# from CNN_Blocks import CustomCNN_01, CustomCNN_02, CustomCNN_03, CustomCNN_04
# from transformers import ViTModel

# class ModifiedViT(nn.Module):
#     def __init__(self, vit_model):
#         super(ModifiedViT, self).__init__()
#         self.vit = vit_model
        
#     def forward(self, x):
#         outputs = self.vit(x)
#         return outputs.last_hidden_state

  
# # Combine the modified ViT and custom CNN
# class ViTWithCustomCNN(nn.Module):
#     def __init__(self, vit_model, cnn_model_block, cnn_layers, hidden_channels, num_classes=2):
#         super(ViTWithCustomCNN, self).__init__()
#         self.modified_vit = ModifiedViT(vit_model)
        
#         if cnn_model_block == "CustomCNN_01":
#             self.custom_cnn = CustomCNN_01(vit_model.config.hidden_size, hidden_channels, cnn_layers, num_classes=num_classes)
#         elif cnn_model_block == "CustomCNN_02":
#             self.custom_cnn = CustomCNN_02(vit_model.config.hidden_size, hidden_channels, cnn_layers, num_classes=num_classes)
#         elif cnn_model_block == "CustomCNN_03":
#             self.custom_cnn = CustomCNN_03(vit_model.config.hidden_size, hidden_channels, cnn_layers, num_classes=num_classes)
#         elif cnn_model_block == "CustomCNN_04":
#             self.custom_cnn = CustomCNN_04(vit_model.config.hidden_size, hidden_channels, cnn_layers, num_classes=num_classes)
#         else:
#             raise ValueError("Please select a valid CNN block")
        
    
#     def forward(self, x):
#         x = self.modified_vit(x)  # Extract features using ViT. x.shape = [batch_size, 197, 768]
#         x = x.permute(0, 2, 1).view(x.size(0), x.size(2), 1, -1)  # Reshape for CNN. x.shape = [batch_size, 768, 1, 197]
#         x = self.custom_cnn(x)
#         return x

# # Assume vit_model is your pre-trained ViT model instance
# vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", torchscript = True)  # Load your ViT model here
# cnn_model_block = "CustomCNN_01"  # Example
# cnn_layers = 4  # Example
# hidden_channels = 768  # Example

# model = ViTWithCustomCNN(vit_model, cnn_model_block, cnn_layers, hidden_channels)
# # print(model.__class__.__name__)
# print(model.modified_vit.__class__.__name__ + model.custom_cnn.__class__.__name__)
# # Script the model
# # scripted_model = torch.jit.script(model)
# scripted_model = torch.jit.trace(model, torch.randn(16, 3, 224, 224))
# # Save the scripted model
# torch.jit.save(scripted_model, 'scripted_vit_with_custom_cnn.pt')

# print("Model has been scripted and saved successfully.")


# import torch

# # import torchvision

# model = torch.jit.load('jit.pt')
# print(model)

# model_jit = torch.jit.script(model)
# torch.jit.save(model_jit, "jit.pt")

# # example = torch.rand(1, 3, 224, 224)
# # traced_script_module = torch.jit.trace(model, example)

# # # Save the TorchScript model
# # traced_script_module.save("traced_resnet_model.pt")
# import torch
# from transformers import AutoImageProcessor, ViTModel
# from PIL import Image
# imgage = "/home/shakib/Work/Personal/caMicroscope/thym/train/til-positive/TCGA-3T-AA9L_0a4b2471c28a9a677752a28579c616ca.png"
# img = Image.open(imgage)
# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", torchscript=True)


# inputs = image_processor(img, return_tensors="pt")


# traced_model = torch.jit.script(model)
# traced_model.save("traced_vit_model.pt")
# print("Model has been traced and saved successfully.")


# import subprocess

# def run_wsinfer(wsi_dir, results_dir, model_path, config_path):
#     command = [
#         'wsinfer', 'run',
#         '--wsi-dir', wsi_dir,
#         '--results-dir', results_dir,
#         '--model-path', model_path,
#         '--config', config_path
#     ]
    
#     result = subprocess.run(command, capture_output=True, text=True)
    
#     if result.returncode != 0:
#         print(f"Error: {result.stderr}")
#     else:
#         print(f"Output: {result.stdout}")

# # Example usage
# wsi_dir = 'slides/'
# results_dir = 'results/'
# model_path = 'jit.pt'
# config_path = 'saved/config.json'

# run_wsinfer(wsi_dir, results_dir, model_path, config_path)
