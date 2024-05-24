import torch
import torch.nn as nn

class DepthDecoder(nn.Module):
    def __init__(self, in_features, final_size):
        super(DepthDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 384, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(384)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(384, 384, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(384)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(384, 1, kernel_size=1)

        self.size3 = final_size
        self.size2 = self.size3[0] // 2, self.size3[1] // 2
        self.size1 = self.size2[0] // 2, self.size2[1] // 2

    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.size1, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = nn.functional.interpolate(x, size=self.size2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = nn.functional.interpolate(x, size=self.size3, mode='bilinear', align_corners=False)
        x = self.conv3(x)

        # Apply ReLU to ensure positive depths
        x = torch.relu(x)
        return x