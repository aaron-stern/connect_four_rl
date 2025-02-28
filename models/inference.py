from models.utils import get_open_columns
import torch
from torch import nn

class ConnectFourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input shape: (batch_size, 6, 7)
        # Define convolutional layers with increasing channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Activation function
        self.relu = nn.ReLU()

        # Flatten the output from convolutional layers
        self.flatten = nn.Flatten()
        
        # Calculate the size of the flattened features
        # After 3 conv layers with padding=1, the spatial dimensions remain the same (6x7)
        # With 64 output channels from conv3, the flattened size is 64*6*7
        flattened_size = 128 * 6 * 7 
        
        # Linear projection to logits (one logit per column for action selection)
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        open_columns = get_open_columns(x.squeeze(1) if x.dim() > 2 else x)
        
        # Process the input through convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # Set invalid moves to negative infinity to ensure they're never selected
        x = x.masked_fill(~open_columns, float('-inf'))
        return x