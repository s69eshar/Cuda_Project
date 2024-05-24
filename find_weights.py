from datasets import CarlaDataset
from torchvision.transforms import v2, InterpolationMode
from collections import Counter
from tqdm import tqdm

root_dir = "/home/nfs/inf6/data/datasets/Carla_Moritz/SyncAngel3/"
transforms = v2.Compose([
    v2.Resize((256, 512), InterpolationMode.BILINEAR, antialias=False),
    v2.ToDtype(torch.float, scale=True),
    #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = CarlaDataset(root_dir, transform=transforms, split='test')

overall_counter = Counter()

for _, sample in tqdm(dataset):
    overall_counter += Counter(sample.flatten().numpy())


weights = []
for key in range(22):
    weight = sum(overall_counter.values()) / (len(overall_counter) * overall_counter[key]) if overall_counter[key] != 0 else 1
    weights.append(weight)

print(weights)