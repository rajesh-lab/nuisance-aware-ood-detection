# From https://github.com/deeplearning-wisc/Spurious_OOD/blob/pub/models/resnet.py
# Commit 4fe49c64ca9693004259e981d151ee0a550a3cdc
# Additional code from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torchvision

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class SimpleClassifier(nn.Module):
    def __init__(self, img_side=28, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.linear = nn.Linear(512, num_classes)  # for img_side=28
        # self.linear = nn.Linear((img_side - 6 * 4)**2 * 32, num_classes)
    
    def forward(self, x):
        out = self.network(x)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return out, y


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))   
        t = self.downsample(x)
        out += t        
        out = F.relu(out)
        
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, orig=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if orig:
            # for large input size
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        else:
            self.conv1 = conv3x3(3,64)
                        
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # end 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out= self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return out, y
        
        # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out= self.maxpool(out)
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        out= self.maxpool(out)
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out

    def load(self, path="resnet_svhn.pth"):
        tm = torch.load(path, map_location="cpu")        
        self.load_state_dict(tm)
    

class ResNetFractalDB(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetFractalDB, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(x.shape)
        # x = self.avgpool(x)
        out = x.view(x.size(0), -1)
        x = self.fc(out)

        return out, x

def load_model(orig=True, num_classes=2):
    '''
    load resnet18
    '''
    torch_model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, orig=orig)
    model_dict = torch_model.state_dict()
    arch = "resnet18"
    pretrained_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                            progress=True)
    # 1. filter out unnecessary keys
    if orig:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "conv1" not in k and k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    torch_model.load_state_dict(model_dict)
    print("ResNet Loading Done")
    return torch_model


class CriticModel(nn.Module):
    def __init__(self, img_side, num_classes, model_arch):
        super().__init__()
        self.y_rep_network = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU()
        )
        if model_arch == "resnet18":
            self.z_rep_network = load_model(orig=img_side==224)
        else:
            self.z_rep_network = SimpleClassifier(img_side=img_side, num_classes=num_classes)
        self.final_block = nn.Sequential(
            nn.Conv1d(3, 1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Linear(127, 2)
        )
    
    def forward(self, rx, y, z):
        """
        Concatenates 512-dim representations for each of x, y, z.
        Result is [batch, 3, 512].
        Then 

        rx: [batch, 512]
        y: [batch, 1]
        z: [batch, 3, 224, 224]

        """
        y_rep = self.y_rep_network(y)  # [batch, 512]
        z_rep, _ = self.z_rep_network(z)  # [batch, 512]
        combined = torch.cat([rx.unsqueeze(1), y_rep.unsqueeze(1), z_rep.unsqueeze(1)], dim=1)  # [batch, 3, 512]
        return self.final_block(combined).squeeze(1)


class CriticModelExact(nn.Module):
    def __init__(self, img_side, num_classes, model_arch):
        super().__init__()
        self.final_block = nn.Sequential(
            nn.Linear(513, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, rx, y):
        """
        Predict z from rx, y
        Result is [batch, 3, 512].
        Then 

        rx: [batch, 512]
        y: [batch, 1]

        """
        combined = torch.cat([rx, y], dim=1)  # [batch, 513]
        return self.final_block(combined)



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            mappings = {
                "fc.weight": "linear.weight",
                "fc.bias": "linear.bias",
            }
            state_dict = {(mappings[k] if k in mappings else k): v for k, v in state_dict.items()}
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50_fractaldb(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    # model = ResNetFractalDB(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = ResNetFractalDB(Bottleneck, [3, 4, 6, 3], num_classes=10000, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def get_model(args):
    import os
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if not args.random_init:
        if args.fractal_db:
            # model_file = "/scratch/lhz209/nood/data/FractalDB-1000_resnet50_epoch90.pth"
            model_file =  "/scratch/lhz209/nood/data/FractalDB-10000_resnet50_epoch90.pth"
        elif args.model_file is not None:
            model_file = args.model_file
        else:
            directory = "/scratch/lhz209/nood/nuisance_ood/src/supervised/checkpoints/{in_dataset}/{name}/{exp}/".format(
                in_dataset=args.in_dataset, name=args.project_name, exp=args.exp_name)
            model_file = os.path.join(directory, f"checkpoint_main.pth.tar")
            print(model_file)
    if args.in_dataset == "cifar10":
        num_classes = 10
        img_side = 32
    elif args.in_dataset in ["cmnist"]:
        num_classes = 2
        img_side = 28
    elif args.in_dataset in ["cxr-small"]:
        num_classes = 2
        img_side = 32
    elif args.in_dataset in ["waterbird", "cxr", "celeba", "cxr-bal"]:
        num_classes = 2
        img_side = 224
    else:
        raise NotImplementedError(f"In dataset not supported: {args.in_dataset}")
    
    orig = img_side == 224
    if args.model_arch == "resnet18":
        model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, orig=orig)
        print(model)
    elif args.model_arch == "resnet50_fractaldb":
        model = resnet50_fractaldb()
    elif args.model_arch == "resnet50":
        model = resnet50()
    elif args.model_arch == "resnet101":
        model = torchvision.models.resnet101(pretrained=True)
    else:
        model = SimpleClassifier(img_side=img_side, num_classes=num_classes)
    if not args.random_init:
        if args.fractal_db:
            model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
        elif args.model_file is not None:
            state_dict = torch.load(model_file, map_location=torch.device(device))["model_dict"]
            rename_key = lambda k: k.replace("featurizer.network.", "").replace("classifier", "linear")
            state_dict = {rename_key(key): value for key, value in state_dict.items()
                if not key.startswith("network") and "num_batches_tracked" not in key}
            state_dict = {k: v for k, v in state_dict.items()
                if k.startswith("conv") or k.startswith("bn") or k.startswith("layer") or k.startswith("linear")}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(model_file, map_location=torch.device(device))["state_dict_model"])
    model.to(device)
    return model

if __name__ == "__main__":
    load_model(True)