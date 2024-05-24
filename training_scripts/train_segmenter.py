import torch
from datasets import CarlaDataset
from torchvision.transforms import v2, InterpolationMode
from modules.segmentation_model import SegmentationModel
from modules.identity import Identity
from torch import nn
import trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torch.utils.tensorboard import SummaryWriter
import os
import shutil


TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", "train_segmenter2")
if not os.path.exists(TBOARD_LOGS):
    os.makedirs(TBOARD_LOGS)
shutil.rmtree(TBOARD_LOGS)
writer = SummaryWriter(TBOARD_LOGS)

root_dir = "/home/nfs/inf6/data/datasets/Carla_Moritz/SyncAngel3/"

transforms = v2.Compose([
    v2.Resize((256, 512), InterpolationMode.BILINEAR, antialias=False),
    v2.ToDtype(torch.float, scale=True),
    #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = CarlaDataset(root_dir, transform=transforms, split='train')
valid_dataset = CarlaDataset(root_dir, transform=transforms, split='validation')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=8)

model = SegmentationModel(filter=Identity())
model.to(device)

#weights = torch.as_tensor([0.23, 6.04, 1, 75.00, 6.03, 62.70, 0.29, 0.49, 0.54, 0.45, 4.33, 85.95,
#                           0.30, 20.72, 1, 1, 69.55, 34.69, 7.879, 29.00, 1, 5.89], device=device)
criterion = nn.CrossEntropyLoss(ignore_index=255)  # Important to ignore 255

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

trainer.train_model(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, 
                    train_loader=train_loader, valid_loader=valid_loader, num_epochs=20, writer=writer, device=device)

print('Training_complete')



