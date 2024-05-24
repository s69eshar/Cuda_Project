import imageio
import numpy as np
from visualizations.segmentation_utils import draw_segmentation_map

def save_im_seg_dep_cam_together(sequence, segs, depths, positions, path):
    images = []
    for i in range(len(sequence)):
        image = (image.permute(1,2,0).numpy() * 255).astype(np.uint8)
        images.append(image)
    imageio.mimsave(path, images)

def save_tensors_as_gif(sequence, path, segmentation=False):
    images = []
    for image in sequence:
        if segmentation:
            image = image.softmax(dim=-3)
            image = draw_segmentation_map(image)
            images.append(image)
        else:
            images.append((image.cpu().permute(1,2,0).squeeze().numpy() * 255).astype(np.uint8))
    imageio.mimsave(path, images)


def save_segs_as_gif(sequence, path):
    images = []
    for image in sequence:
        print((image.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8).shape)
        images.append((image.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8))
    imageio.mimsave(path, images)