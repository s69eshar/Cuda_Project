import torch

@torch.no_grad
def process_video(sequence, model, ego_motion, device):
    pred_segs, pred_depths, pred_transforms = [], [], []
    feat_state = camera_state = last_embedding = None
    for i in range(sequence[0].shape[1]):
        images = sequence[:,i,...].to(device)

        pred_seg, pred_depth, feat_state, embedding = model(images, feat_state)
        pred_segs.append(pred_seg)
        pred_depths.append(pred_depth)


        # We only do the movement calculation if we have two frame of information
        if last_embedding is not None:
            # Model takes in two images and the initial position and returns a transition
            transform, camera_state = ego_motion(last_embedding, embedding, camera_state)
            pred_transforms.append(transform)

        last_embedding = embedding
    return torch.stack(pred_segs, 1), torch.stack(pred_depths, 1), torch.stack(pred_transforms, 1)

@torch.no_grad
def process_video_framewise(sequence, model, device):
    pred_segs = []
    for i in range(sequence[0].shape[1]):
        images = sequence[:,i,...].to(device)
        pred_seg = model(images)
        pred_segs.append(pred_seg)
    return torch.stack(pred_segs, 1)