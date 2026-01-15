import torch
import torch.nn as nn

class QST_Model(nn.Module):
    def __init__(self):
        super(QST_Model, self).__init__()
        
        # Increasing layer width to 128 neurons for better pattern recognition
        self.network = nn.Sequential(
            nn.Linear(3, 128),     # Input: X, Y, Z
            nn.ReLU(),
            nn.Linear(128, 128),   # Expanded hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),    # Tapering down for smooth output transition
            nn.ReLU(),
            nn.Linear(64, 4)       # Output: r00, r11, r01_real, r01_imag
        )

    def forward(self, x):
        return self.network(x)