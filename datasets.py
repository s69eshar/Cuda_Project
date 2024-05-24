import os
import torch
import pickle
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torch.utils.data import Dataset
from torchvision import tv_tensors
import torchvision.transforms.v2.functional as F


class CarlaDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_augmentations=None, split='test', images_per_sequence=20):
        self.root_dir = root_dir
        self.transform = transform
        self.image_augmentations = image_augmentations
        if(split=='train'):
            
            self.towns = [1, 3, 4, 5, 6, 7]
        elif(split=='test'):
            self.towns = [2]
        elif(split=='validation'):
            self.towns = [10]
        else:
            raise ValueError('Excepted split to be train, test, or validation')
        
        self.excluded_sequences = {1 : list(),
                                   2 : list(),
                                   3 : [726, 1171],
                                   4 : list(),
                                   5 : [555],
                                   6 : list(),
                                   7 : list(),
                                   10: list()
        }
        self.images_per_sequence = images_per_sequence
        self.images_per_town = [1500 * self.images_per_sequence  - len(self.excluded_sequences[i]) * self.images_per_sequence  for i in self.towns]
        self.cum_images = [sum(self.images_per_town[:i+1]) for i in range(len(self.images_per_town))]
    def __len__(self):
        excluded_images = sum([len(self.excluded_sequences[town]) for town in self.towns]) * self.images_per_sequence
        return len(self.towns) * 1500 * self.images_per_sequence - excluded_images

    def __getitem__(self, idx):
        town, sequence, image_number = self.get_image_numbers(idx)
        sequence_path = self.get_sequence_path(town, sequence)

        # Read the image and segmentation
        image = self.read_base_image(sequence_path, image_number)
        target = self.read_segmentation(sequence_path, image_number)

        #Apply transformations
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def get_image_numbers(self, idx):
        town_index = [i > idx for i in self.cum_images].index(True)
        town = self.towns[town_index]
        excluded = self.excluded_sequences[town]
        start_indices = [self.cum_images[i-1] if i>0 else 0 for i in range(len(self.cum_images))]
        image_num_in_town = idx - start_indices[town_index]
        sequence = (image_num_in_town) // self.images_per_sequence
        sequence = [x for x in range(sequence + len(excluded) + 1) if x not in excluded][sequence]
        image_number = image_num_in_town % self.images_per_sequence
        return town,sequence,image_number

    def get_sequence_path(self, town, sequence):
        return os.path.join(self.root_dir, f'Town{town:02d}', f'seq_{sequence:04d}')
    
    def read_base_image(self, sequence_path, image_number):
        image_name  = f"img_{image_number:03d}.png"
        image_path = os.path.join(sequence_path, image_name)
        return read_image(image_path, ImageReadMode.RGB)
    
    def read_segmentation(self, sequence_path, image_number):
        image_name  = f"segmentation_{image_number:03d}.png"
        image_path = os.path.join(sequence_path, image_name)
        image = read_image(image_path, ImageReadMode.RGB).long()[0,:,:] - 1
        image[image < 0] = 0
        return tv_tensors.Mask(image)
    
    def read_depth(self, sequence_path, image_number):
        image_name  = f"depth_{image_number:03d}.png"
        image_path = os.path.join(sequence_path, image_name)
        image = F.to_dtype(read_image(image_path, ImageReadMode.RGB), dtype = torch.float, scale=True)
        r, g, b = torch.tensor_split(image, 3, dim=0)
        image = (r / 65536 + g / 256 + b)
        return image

    
    
class MovementDataset(CarlaDataset):
    def __init__(self, root_dir, transform=None, split='test'):
        super().__init__(root_dir, transform=transform, split=split, images_per_sequence=19)

    def __getitem__(self, idx):
        town, sequence, image_number = self.get_image_numbers(idx)
        sequence_path = self.get_sequence_path(town, sequence)

        # Get images
        image1 = self.read_base_image(sequence_path, image_number)
        image2 = self.read_base_image(sequence_path, image_number+1)

        #get camera extrinsics for the frame
        meta_path = os.path.join(sequence_path,'meta.pkl')
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        p1 = torch.as_tensor(metadata['extrinsics'][image_number])
        p2 = torch.as_tensor(metadata['extrinsics'][image_number+1])

        
        if self.transform:
            image1, image2 = self.transform(image1), self.transform(image2)
            
        return image1, image2, p1, p2

class DepthDataset(CarlaDataset):
    def __init__(self, root_dir, transform=None, depth_transform=None, split='test'):
        super().__init__(root_dir, transform=transform, split=split, images_per_sequence=20)
        self.depth_transform = depth_transform

    def __getitem__(self, idx):
        town, sequence, image_number = self.get_image_numbers(idx)
        sequence_path = self.get_sequence_path(town, sequence)

        # Get images
        image = self.read_base_image(sequence_path, image_number)
        depth = self.read_depth(sequence_path, image_number)
        

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        return image, depth

    
class SequenceDataset(CarlaDataset):
    def __init__(self, root_dir, transform=None, image_augmentations=None, split='test'):
        super().__init__(root_dir, transform=transform, image_augmentations=image_augmentations, split=split, images_per_sequence=6)
    

    def __len__(self):
        return sum([1500 - len(self.excluded_sequences[i])  for i in self.towns])

    def __getitem__(self, idx):
        town, sequence, _ = self.get_image_numbers(idx * self.images_per_sequence)
        sequence_path = self.get_sequence_path(town, sequence)

        images, depths, segmentations = [None]*self.images_per_sequence, [None]*self.images_per_sequence, [None]*self.images_per_sequence
        # Get images

        for image_number in range(self.images_per_sequence):
            image = self.read_base_image(sequence_path, image_number)
            depth = self.read_depth(sequence_path, image_number)
            segmentation = self.read_segmentation(sequence_path, image_number)
            if self.transform:
                image, segmentation = self.transform(image, segmentation)
                depth = self.transform(depth)
            
            images[image_number], depths[image_number], segmentations[image_number] = image, depth, segmentation

        #get camera extrinsics for the frame
        meta_path = os.path.join(sequence_path,'meta.pkl')
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        metadata = torch.as_tensor(metadata['extrinsics'])

        images, segmentations, depths = torch.stack(images), torch.stack(segmentations), torch.stack(depths)

        if self. image_augmentations:
                images = self.image_augmentations(images)

        return images, segmentations, depths, metadata

    