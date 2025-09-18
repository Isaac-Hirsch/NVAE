# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import numpy as np
from PIL import Image
import random
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler
from scipy.io import loadmat
import os
import urllib
from lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN

from scripts.identBox.identBoxDataset import IdentBoxDataset, data_transforms_identbox, \
    ConceptsIdentBoxDataset, ConceptsIdentBoxSampler
import torch.nn as nn
import pandas as pd


class StackedMNIST(dset.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(StackedMNIST, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        index1 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index2 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index3 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        self.num_images = 2 * len(self.data)

        self.index = []
        for i in range(self.num_images):
            self.index.append((index1[i], index2[i], index3[i]))

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img = np.zeros((28, 28, 3), dtype=np.uint8)
        target = 0
        for i in range(3):
            img_, target_ = self.data[self.index[index][i]], int(self.targets[self.index[index][i]])
            img[:, :, i] = img_
            target += target_ * 10 ** (2 - i)

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class ConceptsMNIST(Dataset):
    """
    ConceptsMNIST is a extension of MNIST dataset.
    It has 7 concepts each of which is a replication of MNIST with a transformation
    applied to the original MNIST images.
    The transformations are:
        obs:    original MNIST
        scaled: scaled MNIST
        shear:  sheared MNIST
        shift:  shifted MNIST
        swel:   swelled MNIST
        thic:   thickened MNIST
        thin:   thinned MNIST
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(ConceptsMNIST, self).__init__()
        concepts = ("obs", "scaled", "shear", "shift", "swel", "thic", "thin")
        csvs = [pd.read_csv(os.path.join(root, f"normalized_mnist_{concept}.csv"), header=None) for concept in concepts]
        uf = nn.Unflatten(-1, (1, 28, 28))
        data_dict = {concept: uf(torch.tensor(d.values[:, :-2], dtype=torch.float32)) for d, concept in zip(csvs, concepts)}

        train_frac = 0.8
        train_size = int(train_frac * len(data_dict["obs"]))
        
        np.random.seed(0)
        train_indices = np.random.choice(len(data_dict["obs"]), size=train_size, replace=False)
        test_indices = np.setdiff1d(np.arange(len(data_dict["obs"])), train_indices)
        if train:
            self.data = {concept: data_dict[concept][train_indices] for concept in concepts}
            self.data_size = train_size
        else:
            self.data = {concept: data_dict[concept][test_indices] for concept in concepts}
            self.data_size = len(data_dict["obs"]) - train_size
        self.transform = transform

    def __getitem__(self, index):
        concept = None
        match index // self.data_size:
            case 0:
                concept = "obs"
            case 1:
                concept = "scaled"
            case 2:
                concept = "shear"
            case 3:
                concept = "shift"
            case 4:
                concept = "swel"
            case 5:
                concept = "thic"
            case 6:
                concept = "thin"
            case _:
                raise ValueError(f"Invalid index {index} for ConceptsMNIST")
        img = self.data[concept][index % self.data_size]
        if self.transform is not None:
            img = self.transform(img)

        return img, concept
    
    def __len__(self):
        return self.data_size * 7

class ConceptsMNISTSampler(Sampler):
    def __init__(self, dataset, num_concepts, batch_size, args):
        self.batch_size = batch_size
        self.total_samples = len(dataset)
        self.samples_per_concept = self.total_samples // num_concepts
        self.concepts = [i for i in range(num_concepts)]
        self.dataset = dataset
        self.args = args

    def __iter__(self):
        batches = []
        for concept in self.concepts:
            start = concept * self.samples_per_concept
            stop = start + self.samples_per_concept
            indicies = np.arange(start=start, stop=stop, dtype=int)
            np.random.shuffle(indicies)
            for i in range(0, self.samples_per_concept, self.batch_size):
                batch = indicies[i:i + self.batch_size]
                batches.append(batch)
        np.random.shuffle(batches)
        for i, batch in enumerate(batches):
            if i % self.args.global_size == self.args.global_rank:
                yield batch
        
    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size

class ConceptsCeleba(Dataset):
    def __init__(self, root, split: str='train', transform=None,
                 download=False, concepts=None):
        assert split in ['train', 'valid', 'test', 'all']
        assert concepts[0] == 'obs'
        super(ConceptsCeleba, self).__init__()
        self.data = dset.celeba.CelebA(root=root, split=split, target_type='attr', download=download, transform=transform)

        df = pd.read_csv(os.path.join(root, 'celeba', 'list_attr_celeba.txt'), sep='\s+', skiprows=1)
        df_split = pd.read_csv(os.path.join(root, 'celeba', 'list_eval_partition.txt'), sep='\s+', names=['split'], header=None)
        df = pd.merge(df, df_split, left_index=True, right_index=True)
        if split == 'train':
            df = df[df['split'] == 0]
        elif split == 'valid':
            df = df[df['split'] == 1]
        elif split == 'test':
            df = df[df['split'] == 2]

        df = df.drop(['split'], axis=1)
        df.reset_index(inplace=True, drop=True)

        self.concept_indices = {}
        self.concepts = concepts

        all_indicies = {i for i in range(len(self.data))}
        used_indicies = set()

        for concept in concepts[1:]:
            concept_df = df[concept] == 1
            self.concept_indices[concept] = concept_df[concept_df].index.tolist()
            used_indicies = used_indicies.union(set(self.concept_indices[concept]))
        
        self.concept_indices['obs'] = list(all_indicies - used_indicies)
        
        self.length = sum(len(v) for v in self.concept_indices.values())

    def __getitem__(self, index):
        images_seen = 0
        for concept, indicies in self.concept_indices.items():
            if index < images_seen + len(indicies):
                img, target = self.data[indicies[index - images_seen]]
                return img, concept
            images_seen += len(indicies)
        raise IndexError(f"Index {index} out of range for Concepts_Celeba_64 dataset")
    
    def __len__(self):
        return self.length

class ConceptsCelebaNoOversampled(Dataset):
    def __init__(self, root, split: str='train', transform=None,
                 download=False, concepts=None):
        assert split in ['train', 'valid', 'test', 'all']
        assert concepts[0] == 'obs'
        super(ConceptsCelebaNoOversampled, self).__init__()
        self.data = dset.celeba.CelebA(root=root, split=split, target_type='attr', download=download, transform=transform)

        df = pd.read_csv(os.path.join(root, 'celeba', 'list_attr_celeba.txt'), sep='\s+', skiprows=1)
        df_split = pd.read_csv(os.path.join(root, 'celeba', 'list_eval_partition.txt'), sep='\s+', names=['split'], header=None)
        df = pd.merge(df, df_split, left_index=True, right_index=True)
        if split == 'train':
            df = df[df['split'] == 0]
        elif split == 'valid':
            df = df[df['split'] == 1]
        elif split == 'test':
            df = df[df['split'] == 2]

        df = df.drop(['split'], axis=1)
        df.reset_index(inplace=True, drop=True)

        self.concept_indices = {}
        self.concepts = concepts

        all_indicies = {i for i in range(len(self.data))}
        used_indicies = set()

        for concept in concepts[1:]:
            concept_df = df[concept] == 1
            self.concept_indices[concept] = concept_df[concept_df].index.tolist()
            used_indicies = used_indicies.union(set(self.concept_indices[concept]))
        
        self.concept_indices['obs'] = list(all_indicies - used_indicies)

        concept_sets = {concept: set(self.concept_indices[concept]) for concept in concepts}
        for idx in range(len(self.data)):
            seen_in = []
            for concept in concepts[1:]:
                if idx in concept_sets[concept]:
                    seen_in.append(concept)
            if len(seen_in) > 1:
                keep_in = random.choice(seen_in)
                for concept in seen_in:
                    if concept != keep_in:
                        concept_sets[concept].remove(idx)
        
        for concept in concepts:
            self.concept_indices[concept] = sorted(list(concept_sets[concept]))
        
        self.length = sum(len(v) for v in self.concept_indices.values())

    def __getitem__(self, index):
        images_seen = 0
        for concept, indicies in self.concept_indices.items():
            if index < images_seen + len(indicies):
                img, target = self.data[indicies[index - images_seen]]
                return img, concept
            images_seen += len(indicies)
        raise IndexError(f"Index {index} out of range for Concepts_Celeba_64 dataset")
    
    def __len__(self):
        return self.length

class ConceptsCelebaSampler(Sampler):
    def __init__(self, dataset, batch_size, args):
        self.batch_size = batch_size
        self.total_samples = len(dataset)
        self.concepts = dataset.concepts
        self.dataset = dataset
        self.args = args
        self.concept_indices = dataset.concept_indices

    def __iter__(self):
        batches = []
        images_seen = 0
        for concept in self.concepts:
            indicies = np.arange(start=images_seen, stop=images_seen + len(self.concept_indices[concept]), dtype=int)
            np.random.shuffle(indicies)
            for i in range(0, len(self.concept_indices[concept]), self.batch_size):
                batch = indicies[i:i + self.batch_size]
                batches.append(batch)
            images_seen += len(indicies)
        np.random.shuffle(batches)
        for i, batch in enumerate(batches):
            if i % self.args.global_size == self.args.global_rank:
                yield batch
    
    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size

class ConceptsMPI3DToy(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        # Load the toy data
        self.data = np.load(os.path.join(self.root, 'mpi3d', 'mpi3d_toy.npz'))['images']

        # Factors:
        # dim 0: object_color
        # dim 1: object_shape
        # dim 2: object_size
        # dim 3: camera_height
        # dim 4: background_color
        # dim 5: horizontal_axis
        # dim 6: vertical_axis
        self.data = self.data.reshape([6, 6, 2, 3, 3, 40, 40, 64, 64, 3])
        
        self.concepts = ['obs', 'object_color', 'object_shape', 'object_size']

        self.data_dict = {}
        self.data_dict['obs'] = self.data[:3, :3, 0]
        self.data_dict['object_color'] = self.data[3:, :3, 0]
        self.data_dict['object_shape'] = self.data[:3, 3:, 0]
        self.data_dict['object_size'] = self.data[:3, :3, 1]

        for key, value in self.data_dict.items():
            self.data_dict[key] = value.reshape(-1, 64, 64, 3)


        self.concept_len = self.data_dict['obs'].shape[0]  # Number of samples per concept

        np.random.seed(0)
        for concept in self.concepts:
            train_indicies = np.random.choice(self.concept_len, size=int(0.8 * self.concept_len), replace=False)
            test_indicies = np.setdiff1d(np.arange(self.concept_len), train_indicies)

            if self.train:
                self.data_dict[concept] = self.data_dict[concept][train_indicies]
            else:
                self.data_dict[concept] = self.data_dict[concept][test_indicies]
        
        self.concept_len = self.data_dict[self.concepts[0]].shape[0]
        
        self.length = sum(self.data_dict[concept].shape[0] for concept in self.concepts)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of range for dataset.")

        concept = self.concepts[idx // self.concept_len]
        image = self.data_dict[concept][idx % self.concept_len]

        if self.transform:
            image = self.transform(image)
        
        return image, concept
    
class ConceptsMPI3DToyNew(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        # Load the toy data
        self.data = np.load(os.path.join(self.root, 'mpi3d', 'mpi3d_toy.npz'))['images']

        # Factors:
        # dim 0: object_color
        # dim 1: object_shape
        # dim 2: object_size
        # dim 3: camera_height
        # dim 4: background_color
        # dim 5: horizontal_axis
        # dim 6: vertical_axis
        self.data = self.data.reshape([6, 6, 2, 3, 3, 40, 40, 64, 64, 3])
        
        self.concepts = ['obs', 'camera_height', 'object_size']

        self.data_dict = {}
        self.data_dict['obs'] = self.data[:, :, 0, 0]
        self.data_dict['camera_height'] = self.data[:, :, 0, 2]
        self.data_dict['object_size'] = self.data[:, :, 1, 0]

        for key, value in self.data_dict.items():
            self.data_dict[key] = value.reshape(-1, 64, 64, 3)


        self.concept_len = self.data_dict['obs'].shape[0]  # Number of samples per concept

        np.random.seed(0)
        for concept in self.concepts:
            train_indicies = np.random.choice(self.concept_len, size=int(0.8 * self.concept_len), replace=False)
            test_indicies = np.setdiff1d(np.arange(self.concept_len), train_indicies)

            if self.train:
                self.data_dict[concept] = self.data_dict[concept][train_indicies]
            else:
                self.data_dict[concept] = self.data_dict[concept][test_indicies]
        
        self.concept_len = self.data_dict[self.concepts[0]].shape[0]
        
        self.length = sum(self.data_dict[concept].shape[0] for concept in self.concepts)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of range for dataset.")

        concept = self.concepts[idx // self.concept_len]
        image = self.data_dict[concept][idx % self.concept_len]

        if self.transform:
            image = self.transform(image)
        
        return image, concept

class ConceptsMPI3DToySampler(Sampler):
    def __init__(self, dataset, batch_size, args):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_concept = dataset.concept_len
        self.args = args

        self.length = ((self.samples_per_concept + self.batch_size - 1) // self.batch_size) * len(dataset.concepts)
    
    def __iter__(self):
        batches = []
        start = 0
        for concept in self.dataset.concepts:
            concept_indices = list(range(start, start + self.samples_per_concept))
            np.random.shuffle(concept_indices)
            
            for i in range(0, self.samples_per_concept, self.batch_size):
                batch = concept_indices[i:i + self.batch_size]
                batches.append(batch)
            
            start += self.samples_per_concept
            
        np.random.shuffle(batches)  
        for i, batch in enumerate(batches):
            if i % self.args.global_size == self.args.global_rank:
                yield batch
    
    def __len__(self):
        return self.length

def dict_collate_fn(batch):
    data = torch.stack(
        [item[0] for item in batch]
    )
    key = batch[0][1]  # All keys in the batch are the same
    return data, key

class Binarize(object):
    """ This class introduces a binarization transformation
    """
    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args)

def download_omniglot(data_dir):
    filename = 'chardata.mat'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    url = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat'

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        print('Downloaded', filename)

    return


def load_omniglot(data_dir):
    download_omniglot(data_dir)

    data_path = os.path.join(data_dir, 'chardata.mat')

    omni = loadmat(data_path)
    train_data = 255 * omni['data'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))
    test_data = 255 * omni['testdata'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))

    train_data = train_data.astype('uint8')
    test_data = test_data.astype('uint8')

    return train_data, test_data


class OMNIGLOT(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        d = self.data[index]
        img = Image.fromarray(d)
        return self.transform(img), 0     # return zero as label.

    def __len__(self):
        return len(self.data)

def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'mnist':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_mnist(args)
        train_data = dset.MNIST(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.MNIST(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'stacked_mnist':
        num_classes = 1000
        train_transform, valid_transform = _data_transforms_stacked_mnist(args)
        train_data = StackedMNIST(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = StackedMNIST(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'omniglot':
        num_classes = 0
        download_omniglot(args.data)
        train_transform, valid_transform = _data_transforms_mnist(args)
        train_data, valid_data = load_omniglot(args.data)
        train_data = OMNIGLOT(train_data, train_transform)
        valid_data = OMNIGLOT(valid_data, valid_transform)
    elif dataset.startswith('celeba'):
        if 'concepts' not in dataset:
            if dataset == 'celeba_64':
                resize = 64
                num_classes = 40
                train_transform, valid_transform = _data_transforms_celeba64(resize)
                train_data = dset.celeba.CelebA(root=args.data, split="train", target_type='attr', download=False, transform=train_transform)
                valid_data = dset.celeba.CelebA(root=args.data, split="valid", target_type='attr', download=False, transform=valid_transform)
            elif dataset in {'celeba_256'}:
                num_classes = 1
                resize = int(dataset.split('_')[1])
                train_transform, valid_transform = _data_transforms_generic(resize)
                train_data = LMDBDataset(root=args.data, name='celeba', train=True, transform=train_transform)
                valid_data = LMDBDataset(root=args.data, name='celeba', train=False, transform=valid_transform)
            else:
                raise NotImplementedError
        else:
            resize = 64
            num_classes = 10
            concepts = ['obs', 'Male', 'Black_Hair', 'Blond_Hair', 'Bags_Under_Eyes', 'Mouth_Slightly_Open']
            train_transform, valid_transform = _data_transforms_celeba64(resize)
            if 'no_oversampled' in dataset:
                train_data = ConceptsCelebaNoOversampled(root=args.data, split='train', transform=train_transform, concepts=concepts)
                valid_data = ConceptsCelebaNoOversampled(root=args.data, split='valid', transform=valid_transform, concepts=concepts)
            else:
                train_data = ConceptsCeleba(root=args.data, split='train', transform=train_transform, concepts=concepts)
                valid_data = ConceptsCeleba(root=args.data, split='valid', transform=valid_transform, concepts=concepts)
            if args.arch_flag == 'concepts':
                train_sampler = ConceptsCelebaSampler(train_data, args.batch_size, args)
                valid_sampler = ConceptsCelebaSampler(valid_data, args.batch_size, args)
                train_queue = torch.utils.data.DataLoader(
                    train_data, batch_sampler=train_sampler, collate_fn=dict_collate_fn, pin_memory=True, num_workers=2)
                valid_queue = torch.utils.data.DataLoader(
                    valid_data, batch_sampler=valid_sampler, collate_fn=dict_collate_fn, pin_memory=True, num_workers=2)
                return train_queue, valid_queue, num_classes
    elif dataset.startswith('lsun'):
        if dataset.startswith('lsun_bedroom'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['bedroom_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['bedroom_val'], transform=valid_transform)
        elif dataset.startswith('lsun_church'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['church_outdoor_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['church_outdoor_val'], transform=valid_transform)
        elif dataset.startswith('lsun_tower'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['tower_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['tower_val'], transform=valid_transform)
        else:
            raise NotImplementedError
    elif dataset.startswith('imagenet'):
        num_classes = 1
        resize = int(dataset.split('_')[1])
        assert args.data.replace('/', '')[-3:] == dataset.replace('/', '')[-3:], 'the size should match'
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_data = LMDBDataset(root=args.data, name='imagenet-oord', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=args.data, name='imagenet-oord', train=False, transform=valid_transform)
    elif dataset.startswith('ffhq'):
        num_classes = 1
        resize = 256
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_data = LMDBDataset(root=args.data, name='ffhq', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=args.data, name='ffhq', train=False, transform=valid_transform)
    elif dataset.startswith('identbox'):
        num_classes = 1
        directory = os.path.join(args.data, dataset.split('-')[1])
        resize = int(dataset.split('-')[2])
        train_transform, valid_transform = data_transforms_identbox(resize)
        train_data = IdentBoxDataset(directory, train=True, transform=train_transform)
        valid_data = IdentBoxDataset(directory, train=False, transform=valid_transform)
    elif dataset.startswith('3DIdent_concepts'):
        num_classes = 7
        concepts = ['obs', 'bg', 'obj', 'sl']
        resize = int(dataset.split('-')[1])
        train_transform, valid_transform = data_transforms_identbox(resize)
        train_data = ConceptsIdentBoxDataset(data_dir=args.data,
                                                train=True,
                                                concepts=concepts,
                                                transform=train_transform)
        valid_data = ConceptsIdentBoxDataset(data_dir=args.data,
                                                train=False,
                                                concepts=concepts,
                                                transform=valid_transform)
        if args.arch_flag == 'concepts':
            train_sampler = ConceptsIdentBoxSampler(train_data, args.batch_size, args)
            valid_sampler = ConceptsIdentBoxSampler(valid_data, args.batch_size, args)
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_sampler=train_sampler, collate_fn=dict_collate_fn, pin_memory=True, num_workers=2)
            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_sampler=valid_sampler, collate_fn=dict_collate_fn, pin_memory=True, num_workers=2)
            return train_queue, valid_queue, num_classes
    elif dataset == 'concepts_mnist':
        num_classes = 7
        train_transform, valid_transform = _data_transforms_concepts_mnist(args)
        train_data = ConceptsMNIST(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = ConceptsMNIST(
            root=args.data, train=False, download=True, transform=valid_transform)
        
        if args.arch_flag == 'concepts':
            train_sampler = ConceptsMNISTSampler(train_data, num_classes, args.batch_size, args)
            valid_sampler = ConceptsMNISTSampler(valid_data, num_classes, args.batch_size, args)
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_sampler=train_sampler, collate_fn=dict_collate_fn, pin_memory=True, num_workers=0)
            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_sampler=valid_sampler, collate_fn=dict_collate_fn, pin_memory=True, num_workers=1)
            return train_queue, valid_queue, num_classes
    elif dataset in ['concepts_mpi3d_toy', 'concepts_mpi3d_toy_new']:
        train_transform, valid_transform = _data_transforms_concepts_mpi3d_toy()
        if dataset == 'concepts_mpi3d_toy':
            num_classes = 4
            train_data = ConceptsMPI3DToy(root=args.data, train=True, transform=train_transform)
            valid_data = ConceptsMPI3DToy(root=args.data, train=False, transform=valid_transform)
        elif dataset == 'concepts_mpi3d_toy_new':
            num_classes = 3
            train_data = ConceptsMPI3DToyNew(root=args.data, train=True, transform=train_transform)
            valid_data = ConceptsMPI3DToyNew(root=args.data, train=False, transform=valid_transform)

        if args.arch_flag == 'concepts':
            train_sampler = ConceptsMPI3DToySampler(train_data, args.batch_size, args)
            valid_sampler = ConceptsMPI3DToySampler(valid_data, args.batch_size, args)
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_sampler=train_sampler, collate_fn=dict_collate_fn, pin_memory=True, num_workers=0)
            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_sampler=valid_sampler, collate_fn=dict_collate_fn, pin_memory=True, num_workers=0)
            return train_queue, valid_queue, num_classes

    else:
        raise NotImplementedError

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=0, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)

    return train_queue, valid_queue, num_classes

def get_concepts(args) -> list[str]:
    """
    Get the list of concepts for the ConceptsMNIST dataset.
    """
    if args.dataset == 'concepts_mnist':
        return ['obs', 'scaled', 'shear', 'shift', 'swel', 'thic', 'thin']
    elif args.dataset.startswith('celeba_concepts'):
        return ['obs', 'Male', 'Black_Hair', 'Blond_Hair', 'Bags_Under_Eyes', 'Mouth_Slightly_Open']
    elif args.dataset.startswith('3DIdent_concepts'):
        return ['obs', 'bg', 'obj', 'sl']
    elif args.dataset == 'concepts_mpi3d_toy':
        return ['obs', 'object_color', 'object_shape', 'object_size']
    elif args.dataset == 'concepts_mpi3d_toy_new':
        return ['obs', 'camera_height', 'object_size']
    return []

def _data_transforms_cifar10(args):
    """Get data transforms for cifar10."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transform, valid_transform


def _data_transforms_mnist(args):
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        Binarize(),
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        Binarize(),
    ])

    return train_transform, valid_transform


def _data_transforms_stacked_mnist(args):
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])

    return train_transform, valid_transform

def _data_transforms_concepts_mnist(args):
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        Binarize()
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        Binarize()
    ])

    return train_transform, valid_transform


def _data_transforms_generic(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_lsun(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform

def _data_transforms_concepts_mpi3d_toy():
    """Create data transforms for MPI3D toy dataset."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_transform, valid_transform
