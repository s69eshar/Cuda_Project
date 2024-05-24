import torch
import torch.nn as nn

class CameraHead(nn.Module):
    """ 
    Module described in Temporal Motion Integration from section 3.2 of Functionally Modular and Interpretable Temporal 
    Filtering for Robust Segmentation 
    """
    def __init__(self):
        super(CameraHead, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        # Splitting output into translation and rotation parts
        translation = x[:, :3]  # First three elements for translation
        rotation_sinus = torch.clamp(x[:, 3:], min=-1, max=1)  # Clipping rotation sinus values
        
        return translation, rotation_sinus