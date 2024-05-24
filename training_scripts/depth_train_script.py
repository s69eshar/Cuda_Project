import torch
from datasets import DepthDataset
from torchvision.transforms import v2, InterpolationMode
from modules.depth_estimation_model import DepthEstimiationModel
from modules.identity import Identity
from torch import nn
import depth_trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torch.utils.tensorboard import SummaryWriter
import os
import shutil


TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", "depth_pretrain")
if not os.path.exists(TBOARD_LOGS):
    os.makedirs(TBOARD_LOGS)
shutil.rmtree(TBOARD_LOGS)
writer = SummaryWriter(TBOARD_LOGS)

root_dir = "/home/nfs/inf6/data/datasets/Carla_Moritz/SyncAngel3/"

transforms = v2.Compose([
    v2.Resize((256, 512), InterpolationMode.BILINEAR, antialias=False),
    v2.ToDtype(torch.float, scale=True),
])

depth_transforms = v2.Compose([
    v2.Resize((256, 512), InterpolationMode.BILINEAR, antialias=False),
])


train_dataset = DepthDataset(root_dir, transform=transforms, depth_transform=depth_transforms, split='train')
valid_dataset = DepthDataset(root_dir, transform=transforms, depth_transform=depth_transforms, split='validation')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=8)


filter = Identity()
model = DepthEstimiationModel(freeze_backbone=False)
model.to(device)


#weights = torch.ones(22, device=device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

depth_trainer.train_model(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, 
                    train_loader=train_loader, valid_loader=valid_loader, num_epochs=20, writer=writer, device=device, save_frequency=1)