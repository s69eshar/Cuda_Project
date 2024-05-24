import torchvision.transforms as transforms
import numpy as np
import torch

label_map = [
               (0, 0, 0),
               (128, 0, 0),
               (0, 128, 0),
               (128, 128, 0),
               (0, 0, 128),
               (128, 0, 128),
               (0, 128, 128),
               (128, 128, 128),
               (64, 0, 0),
               (192, 0, 0),
               (64, 128, 0),
               (192, 128, 0),
               (64, 0, 128),
               (192, 0, 128),
               (64, 128, 128),
               (192, 128, 128),
               (0, 64, 0),
               (128, 64, 0),
               (0, 192, 0),
               (128, 192, 0),
               (0, 64, 128)
]

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def draw_mask(mask):
    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(mask).astype(np.uint8)
    green_map = np.zeros_like(mask).astype(np.uint8)
    blue_map = np.zeros_like(mask).astype(np.uint8)
    
    for label_num in range(0, len(label_map)):
        index = mask == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map
