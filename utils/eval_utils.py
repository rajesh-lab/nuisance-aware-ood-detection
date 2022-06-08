import os
from collections import defaultdict
import multiprocessing
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

from sklearn.covariance import empirical_covariance

from dataset.cub_dataset import WaterbirdDataset
from dataset.cmnist_dataset import ColoredMNIST
from dataset.celeba_dataset import celebADataset
from utils import AverageMeter
from dataset.cmnist_dataset import get_cmnist_dataloader, get_fmnist_dataloader
from dataset.cifar_dataset import get_cifar100_dataloader, get_svhn_dataloader, get_lsun_dataloader
from dataset.celeba_dataset import get_celeba_ood_dataloader


def get_dataloaders(args, exact=False):
    save_dir = "/scratch/lhz209/nood/data/"
    if args.in_dataset == "waterbird":
        dataset_class = WaterbirdDataset
    elif args.in_dataset == "cmnist":
        dataset_class = ColoredMNIST
    elif args.in_dataset == "celeba":
        dataset_class = celebADataset
    else:
        raise ValueError(f"in_dataset not supported: {args.in_dataset}.")
    
    train_dataset = dataset_class(args, "train")
    val_dataset = dataset_class(args, "val")
    test_dataset = dataset_class(args, "test")

    kwargs = {'pin_memory': True, 'num_workers': 8, 'drop_last': True}
    trainloaderIn = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    testloaderIn = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    if args.out_dataset == 'places365':
        scale = 256.0/224.0
        target_resolution = (224, 224)
        large_transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        testsetout = NuisanceDataWrapper(root="/places/ood1/placesbg",
                                    transform=large_transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=multiprocessing.cpu_count())
    elif args.out_dataset == "cmnist-other":
        testloaderOut = get_cmnist_dataloader(args, data_label_correlation=args.data_label_correlation, split="ood", root_dir="/scratch/lhz209/nood/data")
    elif args.out_dataset == "cmnist-other-blue":
        testloaderOut = get_cmnist_dataloader(args, data_label_correlation=args.data_label_correlation, split="ood_blue", root_dir="/scratch/lhz209/nood/data")
    elif args.out_dataset == "cifar100":
        testloaderOut = get_cifar100_dataloader(args, args.data_label_correlation, split="test", root_dir="/scratch/lhz209/nood/data")
    elif args.out_dataset == "svhn":
        testloaderOut = get_svhn_dataloader(args, args.data_label_correlation, split="test", root_dir="/scratch/lhz209/nood/data")
    elif args.out_dataset == "lsun":
        testloaderOut = get_lsun_dataloader(args, args.data_label_correlation, split="test", root_dir="/scratch/lhz209/nood/data")
    elif args.out_dataset == "fashion_mnist":
        testloaderOut = get_fmnist_dataloader(args, args.data_label_correlation, split="test", root_dir="/scratch/lhz209/nood/data")
    elif args.out_dataset == "celeba-other":
        testloaderOut = get_celeba_ood_dataloader(args)
    else:
        raise NotImplemented(f"{args.out_dataset}")


    return trainloaderIn, testloaderIn, testloaderOut
    

class NuisanceDataWrapper(torchvision.datasets.ImageFolder):
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, 0


def train_mahalanobis(dataloader, model, n_classes):
    means_dict = {i: AverageMeter() for i in range(n_classes)}
    cov = AverageMeter()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    features_by_class = defaultdict(list)
    # estimate mean, streaming
    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        features, _ = model(inputs)  # [batch, features]
        for activation, target in zip(features, targets):
            means_dict[target.item()].update(activation.data, 1)
    # estimate cov, streaming
    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        features, _ = model(inputs)  # [batch, features]
        for activation, target in zip(features, targets):
            centered_activation = activation - means_dict[target.item()].avg
            cov.update(torch.outer(centered_activation, centered_activation).data)
    sample_means = {k: v.avg.to(device) for k, v in means_dict.items()}
    sample_cov = cov.avg
    return sample_means, sample_cov.to(device)

def process_mahalanobis(dataloader, model, sample_means, sample_cov):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    all_scores = []
    for inputs, _, _ in dataloader:
        inputs = inputs.to(device)
        features, _ = model(inputs)
        all_distances = []
        for _, v in sample_means.items():
            centered_data = (features - v).to(device)
            distances = (centered_data @ torch.inverse(sample_cov) * centered_data).sum(dim=1)  # [batch] 
            assert distances.shape == torch.Size([len(inputs)]), distances.shape
            all_distances.append(distances.unsqueeze(0))  # [class, batch]
        all_distances = torch.cat(all_distances, dim=0)
        assert all_distances.dim() == 2 and all_distances.shape[1] == len(inputs), all_distances.shape
        min_distances, _ = torch.min(all_distances, dim=0)
        all_scores.append(min_distances.detach().cpu().numpy())
    return np.concatenate(all_scores, axis=0)


