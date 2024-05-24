#Imports, set cuda device, configure tensor board

import torch
from datasets import MovementDataset
from torchvision.transforms import v2, InterpolationMode
from modules.ego_motion_filter_no_rnn import EgoMotionFilter
from modules.geometry_filter import GeometryFilter
from modules.identity import Identity
from torch import nn
import motion_trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torch.utils.tensorboard import SummaryWriter
import os
import shutil


TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", "train_egomotion")
if not os.path.exists(TBOARD_LOGS):
    os.makedirs(TBOARD_LOGS)
shutil.rmtree(TBOARD_LOGS)
writer = SummaryWriter(TBOARD_LOGS)

root_dir = "/home/nfs/inf6/data/datasets/Carla_Moritz/SyncAngel3/"

transforms = v2.Compose([
    v2.Resize((256, 512), InterpolationMode.BILINEAR, antialias=False),
    v2.ToDtype(torch.float, scale=True),
])


train_dataset = MovementDataset(root_dir, transform=transforms, split='train')
valid_dataset = MovementDataset(root_dir, transform=transforms, split='validation')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=8)


geometry_filter = GeometryFilter(use_cuda=True).to(device)
geometry_filter.load_depth_weights('models/depth_epoch_6.pth')
geometry_filter.load_segmenter_weights('models/segmenter_epoch_18.pth')
ego_motion = EgoMotionFilter().to(device)
ego_motion.load_pretrained('models/ego_motion_proc_19.pth')

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

motion_trainer.train_model(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, 
                    train_loader=train_loader, valid_loader=valid_loader, num_epochs=20, writer=writer, device=device, save_frequency=1)