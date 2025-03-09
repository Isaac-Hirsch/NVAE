import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from typing import Optional
import os

class IdentBoxDataset(Dataset):
    def __init__(self, data_dir, train: bool=False, transform: Optional[Transform]=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        self.file_path = os.path.join(data_dir, "train" if train else "test")
        self.factors_path = os.path.join(self.file_path, "factors")
        self.samples_path = os.path.join(self.file_path, "samples")

        self.m1_factors = np.load(os.path.join(self.factors_path, "m1/latents.npy"))
        self.m2_factors = np.load(os.path.join(self.factors_path, "m2/latents.npy"))

        self.m1_samples_paths = os.listdir(os.path.join(self.samples_path, "m1"))
        self.m2_samples_paths = os.listdir(os.path.join(self.samples_path, "m2"))


    
    def __getitem__(self, index):
        if index < len(self.m2_samples_paths):
            #factors = self.m1_factors[index]
            file_path = os.path.join(self.samples_path, "m1", self.m1_samples_paths[index])
        else:
            #factors = self.m2_factors[index - len(self.m1_factors)]
            file_path = os.path.join(self.samples_path, "m2", self.m2_samples_paths[index - len(self.m1_factors)])
        image = torchvision.io.decode_image(file_path)[:3, : , :]
        if self.transform:
            image = self.transform(image)
        return image


    def __len__(self):
        return len(self.m1_samples_paths) + len(self.m2_samples_paths)

def data_transforms_identbox(size: int):
    train_transform = torchvision.transforms.Compose([
        v2.Resize(size),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    valid_transform = torchvision.transforms.Compose([
        v2.Resize(size),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, valid_transform

        

