from modules.conv_gru import ConvGRUCell
import torch
import torch.cuda
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from modules.depth_decoder import DepthDecoder



class GeometryFilter(nn.Module):
    """
    Models abstract scene features, such as scene contents and geometry
        Maintains temporal consistency of features
    Two output heads: Semantic Segmentation and Depth

    """
    def __init__(self, use_cuda = False) -> None:
        super().__init__()

        deeplab = deeplabv3_resnet50(num_classes = 22)
        self.backbone = deeplab.backbone
        datatype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.ConvRNN = ConvGRUCell((32, 64), 2048, 2048, (3,3), False, datatype)
        self.segmenter = deeplab.classifier
        self.depth_head = DepthDecoder(2048, (256, 512))


    def forward(self, x, feat_state=None):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features["out"]
        embedding = x.detach()
        feat_state = torch.zeros_like(x) if feat_state is None else feat_state
        x = self.ConvRNN(x, feat_state)
        feat_state = x.detach()
        segmentation = self.segmenter(x)
        segmentation = F.interpolate(segmentation, size=input_shape, mode="bilinear", align_corners=False)

        depth = self.depth_head(x)

        return segmentation, depth, feat_state, embedding
    
    def load_segmenter_weights(self, savepath): 
        checkpoint = torch.load(savepath)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.segmenter.load_state_dict(checkpoint['classifier_state_dict'])
    
    def load_depth_weights(self, savepath):
        checkpoint = torch.load(savepath)
        self.depth_head.load_state_dict(checkpoint['classifier_state_dict'])