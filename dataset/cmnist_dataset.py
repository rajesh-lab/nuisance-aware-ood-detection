# From https://github.com/reiinakano/invariant-risk-minimization/blob/master/colored_mnist.py
import os

import numpy as np
from PIL import Image
import multiprocessing

import torch
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torch.utils.data import DataLoader


def color_grayscale_arr(arr, color='red'):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if color == 'red':
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  elif color == 'green':
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  elif color == 'blue':
    arr = np.concatenate([np.zeros((h, w, 2), dtype=dtype),
                          arr], axis=2)
  return arr


class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, args, split, transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root="datasets/", transform=transform,
                                target_transform=target_transform)

    self.data_label_correlation = args.data_label_correlation
    self.exact = args.exact
    self.split = split
    self.prepare_colored_mnist()
    if args.undersample:
      raise NotImplementedError("undersample not supported for cmnist")
    self.deterministic_label = args.deterministic_label

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.dataset[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    if self.exact:
      if self.return_weight:
        raise NotImplemented("Exact + weights not yet supported")
      else:
        z = np.argmax(img.max(axis=(1,2)))
        return img, target, z
    else:
      z = img.max(axis=(1,2)).repeat(784).reshape((3, 28, 28))
      # reduce noise by forcing the color to be exactly the same
      z[z>=254] = 255
      return img, target, z

  def __len__(self):
    return len(self.dataset)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    print(os.path.join(colored_mnist_dir, f'{self.split}_{self.data_label_correlation}.pt'))
    if os.path.exists(os.path.join(colored_mnist_dir, f'{self.split}_{self.data_label_correlation}.pt')):
      print('Colored MNIST dataset already exists')
      self.dataset = torch.load(os.path.join(colored_mnist_dir, f'{self.split}_{self.data_label_correlation}.pt'))
      return

    print('Preparing Colored MNIST')

    dataset = []

    if self.split in ["train", "val"]:
      mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
      mnist = [(im, label) for im, label in mnist if label in [0, 1]]
      train_len = int(len(mnist) * .8)
      mnist = mnist[:train_len] if self.split == "train" else mnist[train_len:]
    elif self.split == "test":
      mnist = datasets.mnist.MNIST(self.root, train=False, download=True)
      mnist = [(im, label) for im, label in mnist if label in [0, 1]]
    elif self.split == "ood":
      mnist = datasets.mnist.MNIST(self.root, train=False, download=True)
      mnist = [(im, label) for im, label in mnist if label not in [0, 1]]
    elif self.split == "ood_blue":
      mnist = datasets.mnist.MNIST(self.root, train=False, download=True)
      mnist = [(im, label) for im, label in mnist if label not in [0, 1]]
    else:
      raise ValueError(f"Not supported: {self.split}")

    for idx, (im, label) in enumerate(mnist):
      im_array = np.array(im)
  
      # different from original color mnist  
      binary_label = label

      # Flip label with 25% probability
      if not self.deterministic_label:
        if np.random.uniform() < 0.25:
            binary_label = binary_label ^ 1

      # Color the image either red or green according to true label
      color_red = binary_label == 0

      # strongly correlated
      if self.split in ["train", "val"] and np.random.uniform() < 1 - self.data_label_correlation:
        color_red = not color_red
      # balanced
      elif self.split == "test" and np.random.uniform() < .5:
        color_red = not color_red

      if self.split != 'ood_blue':
        color = 'red' if color_red else 'green'
        colored_arr = color_grayscale_arr(im_array, color=color)
      else:
        colored_arr = color_grayscale_arr(im_array, color='blue')

      colored_arr = np.transpose(colored_arr, (2, 0, 1))
      colored_arr = colored_arr.astype(np.float32)
      dataset.append((colored_arr, binary_label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    self.dataset = dataset
    os.makedirs(colored_mnist_dir, exist_ok=True)
    torch.save(dataset, os.path.join(colored_mnist_dir, f'{self.split}_{self.data_label_correlation}.pt'))
  
  def get_label_prior(self):
    mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
    mnist = [(im, label) for im, label in mnist if label in [0, 1]]
    train_len = int(len(mnist) * .8)
    mnist = mnist[:train_len]
    labels, counts = np.unique([label for _, label in mnist], return_counts=True)
    return {l:c/len(mnist) for l, c in zip(labels, counts)}
  

def get_cmnist_dataloader(args, data_label_correlation, split, root_dir="datasets", **kwargs):
    kwargs = {'pin_memory': False, 'num_workers': multiprocessing.cpu_count(), 'drop_last': False}
    dataset = ColoredMNIST(args, split=split)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            **kwargs)
    return dataloader


import os
import copy
import multiprocessing
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

save_dir = "datasets/"
def get_fmnist_dataloader(args, data_label_correlation, split, root_dir="datasets", exact=False):
    kwargs = {'pin_memory': False, 'num_workers': multiprocessing.cpu_count(), 'drop_last': False}
    dataset_class = torchvision.datasets.FashionMNIST
    dir_name = 'fashion_mnist'
    
    class NuisanceDataset(dataset_class):
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            print(img.shape)
            img = img.repeat(3, 1, 1)
            z = copy.deepcopy(img)
            z[2:-2, 2:-2] = 0
            return img, label, z
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = NuisanceDataset(os.path.join(save_dir, dir_name), train=split=="train", download=True, transform=transform)
        
    if split in ["train", "val"]:
        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                **kwargs)
    elif split == "test":
        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                **kwargs)
    return dataloader
