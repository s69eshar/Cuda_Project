import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.segmentation_model import SegmentationModel
from modules.identity import Identity
import math
import visualizations.visualizations as visualizations

def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def save_model(geometry_filter, ego_motion, optimizer, epoch, stats, savepath=None):
    """ Saving model checkpoint """
    
    if(not os.path.exists("models")):
        os.makedirs("models")
    if not savepath:
        savepath = f"models/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': geometry_filter.state_dict(),
        'egomotion_state_dict': ego_motion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return

def load_model(savepath, geometry_filter=None, ego_motion=None, optimizer=None, ):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath)
    if geometry_filter is not None:
        geometry_filter.load_state_dict(checkpoint['model_state_dict'])
    if ego_motion is not None:
        ego_motion.load_state_dict(checkpoint['egomotion_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    del checkpoint
    
    return geometry_filter, ego_motion, optimizer, epoch, stats

def draw_image_tensor(image):
    plt.imshow(image.cpu().permute(1,2,0))

def get_pretrained_resnet(savepath):
    model = SegmentationModel(filter=Identity())
    checkpoint = torch.load(savepath)
    model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    return model.backbone

def save_segmenter(model, optimizer, epoch, stats, savepath=None):
    """ Saving model checkpoint """
    
    if(not os.path.exists("models")):
        os.makedirs("models")
    if not savepath:
        savepath = f"models/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'backbone_state_dict': model.backbone.state_dict(),
        'classifier_state_dict': model.classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return

def save_depth_classifier(model, optimizer, epoch, stats, savepath=None):
    """ Saving model checkpoint """
    
    if(not os.path.exists("models")):
        os.makedirs("models")
    if not savepath:
        savepath = f"models/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'classifier_state_dict': model.classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return

def IoU(pred, target, num_classes):
    """ Computing the IoU for a single image """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for lbl in range(num_classes):
        pred_inds = pred == lbl
        target_inds = target == lbl
        
        intersection = (pred_inds[target_inds]).long().sum().cpu()
        union = pred_inds.long().sum().cpu() + target_inds.long().sum().cpu() - intersection
        iou = intersection / (union + 1e-8)
        iou = iou + 1e-8 if union > 1e-8 and not math.isnan(iou) else 0
        ious.append(iou)
    return torch.tensor(ious)

@torch.no_grad()
def add_visualization(model, eval_dataset, epoch, writer, device):
    """ """
    imgs, lbls, preds = [], [], []

    indices = torch.randperm(len(eval_dataset))[:5]
    for i in indices:
        img = torch.unsqueeze(eval_dataset[i][0], 0).to(device)
        lbl = torch.unsqueeze(eval_dataset[i][1], 0)
        print(lbl.shape)
        lbl[lbl == 255] = 0

        outputs = model(img)   
        predicted_class = torch.argmax(outputs, dim=1)
        
        imgs.append(img.cpu())
        lbls.append(lbl.cpu())
        preds.append(predicted_class.cpu())
    imgs = torch.cat(imgs, dim=0)
    lbls = torch.cat(lbls, dim=0)
    preds = torch.cat(preds, dim=0)

    fig, ax = visualizations.qualitative_evaluation(imgs, lbls, preds)
    writer.add_figure("Qualitative Eval", fig, global_step=epoch)
    return