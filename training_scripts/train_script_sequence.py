#Imports, set cuda device, configure tensor board

import torch
from datasets import SequenceDataset
from torchvision.transforms import v2, InterpolationMode
from modules.ego_motion_filter import EgoMotionFilter
from modules.geometry_filter import GeometryFilter
import sequence_trainer
import augmentations
import util

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torch.utils.tensorboard import SummaryWriter
import os
import shutil


TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", "sequence_train_finetune")
if not os.path.exists(TBOARD_LOGS):
    os.makedirs(TBOARD_LOGS)
shutil.rmtree(TBOARD_LOGS)
writer = SummaryWriter(TBOARD_LOGS)

root_dir = "/home/nfs/inf6/data/datasets/Carla_Moritz/SyncAngel3/"

transforms = v2.Compose([
    v2.Resize((256, 512), InterpolationMode.BILINEAR, antialias=False),
    v2.ToDtype(torch.float, scale=True),
])

image_augmentations = v2.Compose([
    augmentations.AddGaussianNoise(0.05),
    v2.ColorJitter(),
    augmentations.change_lighting(),
    augmentations.RandomAddClutter(p=0.25, scale=(0.02,0.1))
])


train_dataset = SequenceDataset(root_dir, transform=transforms, image_augmentations=image_augmentations, split='train')
valid_dataset = SequenceDataset(root_dir, transform=transforms, split='validation')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, drop_last=True)


geometry_filter = GeometryFilter(use_cuda=True).to(device)
geometry_filter.load_depth_weights('models/depth_epoch_15.pth')
geometry_filter.load_segmenter_weights('models/segmenter_epoch_10.pth')
ego_motion_filter = EgoMotionFilter().to(device)
ego_motion_filter.load_pretrained('models/ego_motion2_19_proc.pth')

for param in geometry_filter.backbone.parameters():
    param.requires_grad = False

for param in geometry_filter.segmenter.parameters():
    param.requires_grad = False


optimizer = torch.optim.Adam(geometry_filter.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
#geometry_filter, ego_motion_filter, optimizer, epoch, stats = util.load_model('models/sequence_ep_4.pth', geometry_filter=geometry_filter, ego_motion=ego_motion_filter, optimizer=optimizer)

sequence_trainer.train_model(model=geometry_filter, ego_motion=ego_motion_filter, optimizer=optimizer, scheduler=scheduler, 
                    train_loader=train_loader, valid_loader=valid_loader, num_epochs=10, writer=writer, device=device, start_epoch=0)