import numpy as np

import torch
from torch.utils.data import Dataset, Sampler
import torchvision
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from typing import Optional, List
import os
import math

TRAIN_FRAC = 0.8

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

class ConceptsIdentBoxDataset(Dataset):
    def __init__(
                self,
                data_dir,
                train: bool=True,
                concepts: Optional[List[str]]=None,
                transform: Optional[Transform]=None,
                seed: int=1337
            ):
        self.data_list = []
        self.factors_list = []
        self.train = train
        self.transform = transform
        self.concepts = concepts

        np.random.seed(seed=seed)

        for concept in concepts:
            factor_path = os.path.join(data_dir, concept, "raw_latents.npy")
            image_path = os.path.join(data_dir, concept, "images")

            factor = np.load(factor_path)
            self.factors_list.append(factor)

            num_samples = factor.shape[0]
            name_length = int(math.log10(num_samples - 1)) + 1

            images = ["" for _ in range(num_samples)]
            for i in range(num_samples):
                image_name = str(i)
                image_name = "0" * (name_length - len(image_name)) + image_name + ".png"
                images[i] = os.path.join(image_path, image_name)
            train_choice = np.random.choice(a=images, size=int(TRAIN_FRAC * num_samples), replace=False).tolist()
            if train:
                choice = train_choice
            if not train:
                choice = list(set(images) - set(train_choice))
            self.data_list.append(choice)
        
        self.length = sum(map(len, self.data_list))
    
    def __getitem__(self, index):
        seen_samples = 0
        for i, data in enumerate(self.data_list):
            if index < seen_samples + len(data):
                file_path = data[index - seen_samples]
                image = torchvision.io.decode_image(file_path)[:3, : , :]
                if self.transform:
                    image = self.transform(image)
                return image, self.concepts[i]
            seen_samples += len(data)
        raise Exception("Index out of range.")
    
    def __len__(self):
        return self.length

class ConceptsIdentBoxSampler(Sampler):
    def __init__(self, dataset, batch_size, args):
        self.batch_size = batch_size
        self.total_samples = len(dataset)
        self.concepts = dataset.concepts
        self.dataset = dataset
        self.args = args
        self.data_list = dataset.data_list
        self.epoch = 0
        self.total_batches = (self.total_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # Use deterministic seeding so all ranks shuffle identically
        rng = np.random.RandomState(self.epoch + self.args.seed if hasattr(self.args, 'seed') else self.epoch)
        batches = []
        images_seen = 0
        for i, concept in enumerate(self.concepts):
            indicies = np.arange(start=images_seen, stop=images_seen + len(self.data_list[i]), dtype=int)
            rng.shuffle(indicies)
            for j in range(0, len(indicies), self.batch_size):
                batch = indicies[j:j + self.batch_size]
                batches.append(batch)
            images_seen += len(indicies)
        rng.shuffle(batches)
        
        # Pad batches so all ranks get the same number
        total_batches = len(batches)
        world_size = getattr(self.args, 'global_size', 1)
        padded_total = ((total_batches + world_size - 1) // world_size) * world_size
        # Repeat batches to fill padding
        while len(batches) < padded_total:
            batches.append(batches[len(batches) % total_batches])
        
        # Each rank gets every world_size-th batch
        for i, batch in enumerate(batches):
            if i % world_size == getattr(self.args, 'global_rank', 0):
                yield batch
    
    def __len__(self):
        # Return per-rank batch count (padded to be equal across ranks)
        world_size = getattr(self.args, 'global_size', 1)
        return (self.total_batches + world_size - 1) // world_size

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
