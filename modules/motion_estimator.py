from torch import nn
import torch

class MotionEstimator(nn.Module):
    """ 
    Based on the Motion Estimation (f_mot) from section 3.2 of Functionally Modular and Interpretable Temporal 
    Filtering for Robust Segmentation 
    """
    def __init__(self):
        """ Model initializer """
        super().__init__()
        
        # layer 1
        # (4096, 32, 64)
        conv1 = nn.Conv2d(in_channels=4096, out_channels=1024, kernel_size=3, stride=2, padding=1)
        norm1 = nn.BatchNorm2d(1024)
        relu1 = nn.ReLU()
        self.layer1 = nn.Sequential(
                conv1, norm1, relu1
            )
      
        # layer 2
        # (1024, 16, 32)
        conv2 = nn.Conv2d(in_channels=1024, out_channels=256,  kernel_size=3, stride=2, padding=1)
        norm2 = nn.BatchNorm2d(256)
        relu2 = nn.ReLU()
        self.layer2 = nn.Sequential(
                conv2, norm2, relu2
            )
        
        # layer 3
        # (256, 8, 16)
        conv3 = nn.Conv2d(in_channels=256, out_channels=128,  kernel_size=3, stride=2, padding=1)       
        norm3 = nn.BatchNorm2d(128)
        relu3 = nn.ReLU()
        self.layer3 = nn.Sequential(
                conv3, norm3, relu3
            )
        
        # (128, 4, 8)
        self.avg_pooling = nn.AvgPool2d((4,8))

        # (128)
        self.fc1 = nn.Linear(128, 128)
        self.bn = nn.BatchNorm1d(128)
        return
        
    def forward(self, embedding1, embedding2):
        """ Forward pass """
        batch_size = embedding1.shape[0]
        x = torch.cat((embedding1, embedding2), 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pooling(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = x.view(batch_size, -1)
        return x
    
def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params