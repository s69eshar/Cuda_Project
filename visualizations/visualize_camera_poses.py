import os
import numpy as np
import matplotlib as plt
from visualizations.camera_pose_visualizer import CameraPoseVisualizer

def visualize_camera_positions(positions):
    minx = min(pos[0,3] for pos in positions)
    maxx = max(pos[0,3] for pos in positions)
    miny = min(pos[1,3] for pos in positions)
    maxy = max(pos[1,3] for pos in positions)
    minz = min(pos[2,3] for pos in positions)
    maxz = max(pos[2,3] for pos in positions)

    visualizer = CameraPoseVisualizer([minx-1, maxx+1], [miny-1, maxy+1], [minz-1, maxz+1])

    length = len(positions)
    position = np.eye(4)
    visualizer.extrinsic2pyramid(position, 'r', 0.5)
    for i, position in enumerate(positions):
        visualizer.extrinsic2pyramid(position, 'r', 0.5)
    visualizer.show()
    
def visualize_camera_transformations(transforms):
    positions = []
    positions.append(np.eye(4))
    for transform in transforms:
        positions.append(transform @ positions[-1])
    visualize_camera_positions(positions)

