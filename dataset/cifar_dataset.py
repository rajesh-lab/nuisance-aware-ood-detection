import os
import copy
import multiprocessing
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms

save_dir = "datasets"
def get_cifar10_dataloader(args, data_label_correlation, split, root_dir="datasets", exact=False, **kwargs):
    return get_dataloader(args, data_label_correlation, split, root_dir, exact, dataset="CIFAR10")

def get_cifar100_dataloader(args, data_label_correlation, split, root_dir="datasets", exact=False, **kwargs):
    return get_dataloader(args, data_label_correlation, split, root_dir, exact, dataset="CIFAR100")

def get_svhn_dataloader(args, data_label_correlation, split, root_dir="datasets", exact=False, **kwargs):
    return get_dataloader(args, data_label_correlation, split, root_dir, exact, dataset="SVHN")

def get_lsun_dataloader(args, data_label_correlation, split, root_dir="datasets", exact=False, **kwargs):
    return get_dataloader(args, data_label_correlation, split, root_dir, exact, dataset="LSUN")

def get_dataloader(args, data_label_correlation, split, root_dir="datasets", exact=False, dataset="CIFAR10"):
    kwargs = {'pin_memory': False, 'num_workers': multiprocessing.cpu_count(), 'drop_last': False}
    if dataset=="CIFAR10":
        dataset_class = torchvision.datasets.CIFAR10
        dir_name = 'cifar10'
    elif dataset=="CIFAR100":
        dataset_class = torchvision.datasets.CIFAR100
        dir_name = 'cifar100'
    elif dataset=="SVHN":
        dataset_class = torchvision.datasets.SVHN
        dir_name = "svhn"
    elif dataset=="LSUN":
        dataset_class = torchvision.datasets.LSUN
        dir_name = "lsun"
    
    class NuisanceDataset(dataset_class):
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            z = copy.deepcopy(img)
            z[2:-2, 2:-2] = 0
            return img, label, z
    
    if args.in_dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    elif args.in_dataset == "waterbird":
        scale = 256.0/224.0
        target_resolution = (224, 224)
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif args.in_dataset == "celeba":
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        target_resolution = (224, 224)
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif args.in_dataset == "cmnist":
        transform = None
    else:
        raise ValueError("in_dataset transformation not supported")

    if "CIFAR" in dataset:
        dataset = NuisanceDataset(os.path.join(save_dir, dir_name), train=split=="train", download=True, transform=transform)
    elif dataset == "LSUN":
        dataset = NuisanceDataset(os.path.join(save_dir, dir_name), classes=split, transform=transform)
    else:
        dataset = NuisanceDataset(os.path.join(save_dir, dir_name), split=split, download=True, transform=transform)
    num_indices = len(dataset)
    if split == "train":
        sampler = SubsetRandomSampler(range(int(num_indices * .8)))
    elif split == "val":
        sampler = SubsetRandomSampler(range(int(num_indices * .8), num_indices))
        
    if split in ["train", "val"]:
        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                sampler=sampler,
                                **kwargs)
    elif split == "test":
        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                **kwargs)
    return dataloader

def get_cifar10_label_prior(args, data_label_correlation, split, root_dir="datasets", exact=False, **kwargs):
    return {i: 1/10 for i in range(10)}