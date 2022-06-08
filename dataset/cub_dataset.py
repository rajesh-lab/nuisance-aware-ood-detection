# From https://github.com/deeplearning-wisc/Spurious_OOD/blob/pub/datasets/cub_dataset.py
# Commit b6fdd5a09e3ac9ba6bb2fe0c07d765781688c1d3
   
import os
import copy
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import multiprocessing

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class WaterbirdDataset(Dataset):
    def __init__(self, args, split):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.env_dict = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        self.split = split
        self.root_dir  = "datasets"
        self.dataset_name = "waterbird_complete"+"{:0.2f}".format(args.data_label_correlation)[-2:]+"_forest2water2"
        self.dataset_dir = os.path.join(self.root_dir, "waterbirds_erm", self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.') 
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[self.split]]
        self.undersample = args.undersample
        if self.undersample and self.split in ["train", "val"] and (self.split == "val" or not args.undersample_val_only):
            print("undrsampling")
            np.random.seed(0)
            group_counts = self.metadata_df.groupby(["y", "place"]).split.count().to_dict()
            smallest_group = sorted(group_counts, key=group_counts.get)[0]
            lowest_count = group_counts[smallest_group]
            new_df_subsets = []
            for key in group_counts.keys():
                row_idx = self.metadata_df[(self.metadata_df["y"]==key[0])&(self.metadata_df["place"]==key[1])].index
                subset_idx = np.random.choice(row_idx, size=lowest_count, replace=False)
                subset = self.metadata_df.loc[subset_idx]
                new_df_subsets.append(subset)
            self.metadata_df = pd.concat(new_df_subsets)

        self.y_array = self.metadata_df['y'].values
        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = get_transform_cub(self.split=='train')
        self.exact = args.exact

        group_counts = self.metadata_df.groupby(["y", "place"]).split.count().to_dict()
        # weights should be inverse of count proportions
        weights = {k: len(self.metadata_df) / v for k, v in group_counts.items()}
        self.weights = {k: v / sum(weights.values()) for k, v in weights.items()}

    def __len__(self):
        return len(self.filename_array)
    
    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        if self.exact:
            return img, y, place
        else:
            z = copy.deepcopy(img)
            z[:, 14:-14, 14:-14] = 0
            return img, y, z

    def get_label_prior(self):
        label_counts = self.metadata_df.groupby("y").split.count().to_dict()
        return {k: v/len(self.metadata_df) for k, v in label_counts.items()}
    
    def get_nuisance_prior(self):
        nuisance_counts = self.metadata_df.groupby("place").split.count().to_dict()
        return {k: v/len(self.metadata_df) for k, v in nuisance_counts.items()}


    
def get_transform_cub(train):
    scale = 256.0/224.0
    target_resolution = (224, 224)
    assert target_resolution is not None

    # apply the same transform to train, test, val
    transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    if not train:
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform