from typing import Dict
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models import resnet50
from modules.depth_decoder import DepthDecoder
from modules.identity import Identity

class DepthEstimiationModel(nn.Module):
    def __init__(self, resnet50_state_dict = None, freeze_backbone=False) -> None:
        super().__init__()

        #deeplab = deeplabv3_resnet50(num_classes = 22)
        self.backbone = resnet50(replace_stride_with_dilation=[False, True, True])
        if resnet50_state_dict is not None:
            self.backbone.load_state_dict(resnet50_state_dict)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.avgpool = Identity()
        self.backbone.fc = Identity()

        self.classifier = DepthDecoder(2048, (256, 512))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = x.reshape((-1, 2048, 32, 64))
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x