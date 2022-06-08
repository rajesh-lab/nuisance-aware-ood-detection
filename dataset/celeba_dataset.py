# from https://github.com/deeplearning-wisc/Spurious_OOD/blob/pub/datasets/celebA_dataset.py
# commit b6fdd5a09e3ac9ba6bb2fe0c07d765781688c1d3

import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class celebADataset(Dataset):
    def __init__(self, args, split, return_weight=True):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        # (y, gender)
        self.env_dict = {
            (0, 0): 0,   # nonblond hair, female
            (0, 1): 1,   # nonblond hair, male
            (1, 0): 2,   # blond hair, female
            (1, 1): 3    # blond hair, male
        }
        self.split = split
        self.dataset_name = 'celebA'
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'list_attr_celeba.csv')
        )
        split_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'list_eval_partition.csv')
        )
        self.metadata_df = self.metadata_df.merge(split_df, on="image_id")
        self.metadata_df = self.metadata_df[self.metadata_df['partition']==self.split_dict[self.split]]
        self.y_array = self.metadata_df['Blond_Hair'].values
        self.y_array = np.array([0 if el==-1 else el for el in self.y_array])
        self.gender_array = self.metadata_df['Male'].values
        self.gender_array = np.array([0 if el==-1 else el for el in self.gender_array])
        self.filename_array = self.metadata_df['image_id'].values
        self.transform = get_transform_cub(self.split=='train')
        self.balance_correlation_sizes = args.balance_correlation_sizes
        if self.split in ['train', 'val']:
            self.subsample(args.data_label_correlation)
        elif self.split == "test":
            self.subsample(.5)
        # undersample from the already created dataset
        if args.undersample:
            if args.undersample_val_only and self.split == "val":
                self.subsample(.5)
            elif self.split in ["train", "val"]:
                self.subsample(.5)
        assert args.exact, "Need to run exact reweighting"

        from collections import Counter
        group_counts = Counter()
        for y, g in zip(self.y_array, self.gender_array):
            group_counts[(y, g)] += 1
        # weights should be inverse of count proportions
        weights = {k: len(self.y_array) / v for k, v in group_counts.items()}
        self.weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        self.return_weight = return_weight
        
    
    def subsample(self, ratio = 0.6):
        np.random.seed(1)
        train_group_idx = {
            (0, 0): np.array([]),   # nonblond hair, female
            (0, 1): np.array([]),   # nonblond hair, male
            (1, 0): np.array([]),   # blond hair, female
            (1, 1): np.array([])    # blond hair, male
        }
        for idx, (y, gender) in enumerate(zip(self.y_array, self.gender_array)):
            train_group_idx[(y, gender)] = np.append(train_group_idx[(y, gender)],idx)
        
        small_sample_size = len(train_group_idx[(1, 1)])
        big_sample_size = int(ratio/(1-ratio)*len(train_group_idx[(1, 1)]))
        undersampled_idx_00 = np.random.choice(train_group_idx[(0, 0)], small_sample_size, replace=False)
        undersampled_idx_11 = train_group_idx[(1, 1)]
        undersampled_idx_01 = np.random.choice(train_group_idx[(0, 1)], big_sample_size, replace=False)
        undersampled_idx_10 = np.random.choice(train_group_idx[(1, 0)], big_sample_size, replace=False)
        undersampled_idx = np.concatenate( (undersampled_idx_00, undersampled_idx_11, undersampled_idx_01, undersampled_idx_10) )
        undersampled_idx = undersampled_idx.astype(int)

        self.y_array = self.y_array[undersampled_idx]
        self.gender_array = self.gender_array[undersampled_idx]
        self.filename_array = self.filename_array[undersampled_idx]
        print(len(self.y_array))

        if self.balance_correlation_sizes:
            if self.split == "train":
                size = 5548
            elif self.split == "val":
                size = 728
            elif self.split == "test":
                size = 720
            subsetted_idx = np.random.choice(list(range(len(self.y_array))), size, replace=False)
            self.y_array = self.y_array[subsetted_idx]
            self.gender_array = self.gender_array[subsetted_idx]
            self.filename_array = self.filename_array[subsetted_idx]

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        gender = self.gender_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        return img, y, gender
    
    def get_label_prior(self):
        label_prior = {
            0: (self.y_array==0).sum()/len(self.y_array),
            1: (self.y_array==1).sum()/len(self.y_array)
        }
        print(label_prior)
        return label_prior
    
    def get_nuisance_prior(self):
        nuisance_prior = {
            0: (self.gender_array==0).sum()/len(self.gender_array),
            1: (self.gender_array==1).sum()/len(self.gender_array)
        }
        print(nuisance_prior)
        return nuisance_prior



class celebAOodDataset(Dataset):
    def __init__(self):
        self.dataset_name = 'celebA'
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celebA_ood.csv'))

        self.filename_array = self.metadata_df['image_id'].values
        self.transform = get_transform_cub(train=False)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        return img, 0, 0


def get_transform_cub(train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if not train:
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_celeba_dataloader(args, split, **kwargs):
    kwargs = {'pin_memory': True, 'num_workers': 8, 'drop_last': True}
    dataset = celebADataset(args, split)
    dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                **kwargs)
    return dataloader


def get_celeba_label_prior(args, split, return_weight=True, **kwargs):
    dataset = celebADataset(args, split)
    return dataset.get_label_prior()

def get_celeba_nuisance_prior(args, split, return_weight=True, **kwargs):
    dataset = celebADataset(args, split)
    return dataset.get_nuisance_prior()


def get_celeba_ood_dataloader(args, **kwargs):
    kwargs = {'pin_memory': True, 'num_workers': 8, 'drop_last': True}
    dataset = celebAOodDataset()
    dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                **kwargs)
    return dataloader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OOD training for multi-label classification')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64) used for training')
    parser.add_argument('--ood-batch-size', default= 64, type=int,
                    help='mini-batch size (default: 400) used for testing')
    args = parser.parse_args()

    dataloader = get_celeba_dataloader(args, split='train')
    ood_dataloader = get_celeba_ood_dataloader(args)