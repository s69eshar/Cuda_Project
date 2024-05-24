import imageio
import numpy as np

def save_image(image, path):
    imageio.imsave(path, (image.permute(1,2,0).squeeze().numpy() * 255).astype(np.uint8))