# From https://github.com/rajesh-lab/nurd-code/blob/main/code/nrd-xray/dataset_utils.py
# Commit 415d4335c57e0c381b4a5c09dec4e46b834bcab6
import os
import numpy as np
import cv2
import torch
import multiprocessing
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import TensorDataset, DataLoader


def border_pad(image, cfg):
    h, w, c = image.shape

    if cfg.border_pad == 'zero':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=0.0)
    elif cfg.border_pad == 'pixel_mean':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=cfg.pixel_mean)
    else:
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode=cfg.border_pad)

    return image


def fix_ratio(image, cfg):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.long_side
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.long_side
        h_ = round(w_ / ratio)

    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image = border_pad(image, cfg)

    return image


def transform(image, cfg):
    assert image.ndim == 2, "image must be gray image"
    # if cfg.use_equalizeHist:
    #    image = cv2.equalizeHist(image)

    # if cfg.gaussian_blur > 0:
    #     image = cv2.GaussianBlur(
    #         image,
    #         (cfg.gaussian_blur, cfg.gaussian_blur), 0)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = fix_ratio(image, cfg)
    # augmentation for train or co_train

    # normalization
    image = image.astype(np.float32) - cfg.pixel_mean
    # vgg and resnet do not use pixel_std, densenet and inception use.
    if cfg.pixel_std:
        image /= cfg.pixel_std
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    # image = image.transpose((2, 0, 1))

    return image


def Common(image):

    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def Aug(image):
    img_aug = tfs.Compose([
        tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), # no translation
                         scale=(0.95, 1.05))#, fillcolor=128)
    ])
    image = img_aug(image)

    return image


def GetTransforms(image, target=None, type='common'):
    # taget is not support now
    if target is not None:
        raise Exception(
            'Target is not support now ! ')
    # get type
    if type.strip() == 'Common':
        image = Common(image)
        return image
    elif type.strip() == 'None':
        return image
    elif type.strip() == 'Aug':
        image = Aug(image)
        return image
    else:
        raise Exception(
            'Unknown transforms_type : '.format(type))
