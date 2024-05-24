from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

class SegmentationModel(nn.Module):
    def __init__(self, filter: nn.Module) -> None:
        super().__init__()

        deeplab = deeplabv3_resnet50(num_classes = 22)
        self.backbone = deeplab.backbone
        self.filter = filter
        self.classifier = deeplab.classifier

    def forward(self, x: Tensor):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        features = self.filter(features)

        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x