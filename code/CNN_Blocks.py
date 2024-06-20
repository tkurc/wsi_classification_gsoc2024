import torch
import torch.nn as nn

# Custom CNN 01: 2 CNN layers, 1 Linear layer
class CustomCNN_01(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers = 2, num_classes=2):
        super(CustomCNN_01, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        for i in range(num_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x):
        # Ensure the input has 4 dimensions [batch_size, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Adding a channel dimension
        x = self.cnn(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

# Custom CNN 02: 3 CNN layers, 3 Dropout layer, 1 Linear layer
class CustomCNN_02(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers = 3, num_classes=2):
        super(CustomCNN_02, self).__init__()
        layers = []
        self.dropout_rate = 0.3
        layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        for i in range(num_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        self.cnn = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x):
        # Ensure the input has 4 dimensions [batch_size, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Adding a channel dimension
        x = self.cnn(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

# Custom CNN 03: 3 CNN layers, 3 Dropout layer, 3 Batch normalization layer, 2 Linear layer
class CustomCNN_03(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers = 3, num_classes=2):
        super(CustomCNN_03, self).__init__()
        layers = []
        self.dropout_rate = 0.3
        layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        for i in range(num_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.BatchNorm2d(hidden_channels))
        self.cnn = nn.Sequential(*layers)
        self.classifier1 = nn.Linear(hidden_channels, hidden_channels)
        self.classifier2 = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x):
        # Ensure the input has 4 dimensions [batch_size, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Adding a channel dimension
        x = self.cnn(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

# Custom CNN 04: 3 CNN layers, 1 Linear layer, Skip connection
class CustomCNN_04(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=3, num_classes=2):
        super(CustomCNN_04, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ))
        
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x):
        # Ensure the input has 4 dimensions [batch_size, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Adding a channel dimension
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Apply layers with skip connections
        for layer in self.layers:
            residual = x  # Save input for skip connection
            out = layer(x)
            x = out + residual  # Apply skip connection

        # Global average pooling
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

# Custom Model 01
# C_01 = CustomCNN_01(input_channels=768, hidden_channels=64)
# dummy_input = torch.randn(16, 768, 224, 224)
# output_01 = C_01(dummy_input)
# print(output_01.shape)


# Custom Model 02
# C_02 = CustomCNN_02(input_channels=768, hidden_channels=64)
# dummy_input = torch.randn(16, 768, 224, 224)
# output_02 = C_02(dummy_input)
# print(output_02.shape)

# Custom Model 03
# C_03 = CustomCNN_03(input_channels=768, hidden_channels=64)
# dummy_input = torch.randn(16, 768, 224, 224)
# output_03 = C_03(dummy_input)
# print(output_03.shape)

# Custom Model 04
# C_04 = CustomCNN_04(input_channels=768, hidden_channels=64)
# dummy_input = torch.randn(16, 768, 224, 224)
# output_04 = C_04(dummy_input)
# print(output_04.shape)