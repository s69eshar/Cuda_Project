import torch
from torchvision.transforms import v2
from torch import nn
import random
import math

import torchvision.transforms.v2.functional as F

class AddGaussianNoise(nn.Module):
    def __init__(self, std=1.):
        super(AddGaussianNoise, self).__init__()
        self.std = std
        
    def forward(self, tensor):
        return torch.clip(tensor + torch.randn(tensor.size()) * self.std, min=0, max=1)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

class RandomAddClutter(nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        super(RandomAddClutter, self).__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale, ratio, value = None):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_h, img_w = img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w

        # Return original image
        return 0, 0, img_h, img_w, img
    
    def forward(self, images):
        num_images = images.shape[0]

        x, y, h, w = self.get_params(images[0], scale=self.scale, ratio=self.ratio)

        color = torch.mean(images[0], [-2,-1])
        
        images[:,:,x:x+h,y:y+w] = color.reshape(1,-1,1,1).expand(num_images,-1,h,w)

        return images

class change_lighting(nn.Module):
    def __init__(self, probability=1, intensity_std=0.5, decay_factor = 1.1):
        super(change_lighting, self).__init__()
        self.prob = probability
        self.intensity_std = intensity_std
        self.decay_factor = decay_factor

    def forward(self, tensor):
        if self.prob < random.random():
            return tensor
        
        start_i = random.randrange(tensor.shape[0])
        factor = max(0.0,random.gauss(1, self.intensity_std))

        for i in range(start_i, tensor.shape[0]):
            tensor[i] = F.adjust_brightness(tensor[i], factor)
            factor = (factor - 1) / self.decay_factor + 1
        return tensor


