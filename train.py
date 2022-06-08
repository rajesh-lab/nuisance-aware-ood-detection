# From https://github.com/deeplearning-wisc/Spurious_OOD/blob/pub/train_bg.py
# Commit 4fe49c64ca9693004259e981d151ee0a550a3cdc

import argparse
import os
import time
import random
import logging
import itertools
import copy
import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models.resnet import load_model, CriticModel, get_model, SimpleClassifier
from utils import AverageMeter, save_checkpoint, accuracy
from dataset.cub_dataset import WaterbirdDataset
from dataset.cmnist_dataset import ColoredMNIST
import math
from datetime import datetime

parser = argparse.ArgumentParser(description=' use resnet (pretrained)')

parser.add_argument('--in_dataset', default="celebA", type=str, help='in-distribution dataset e.g. IN-9')
parser.add_argument('--model_arch', default='resnet18', type=str, help='model architecture e.g. resnet50')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
# ID train & val batch size
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    help='mini-batch size (default: 64) used for training')
# training schedule
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run, default = 30')
parser.add_argument('--reweight_epochs', default=0, type=int,
                    help='number of total epochs to run reweight model, default = 1')
parser.add_argument('--critic_epochs', default=0, type=int,
                    help='number of total epochs to run reweight model, default = 1')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--optimizer', default="adam", type=str)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.005, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--data_label_correlation', default=0.9, type=float,
                    help='data_label_correlation')
parser.add_argument('--cosine', default=1, type=int, help='using cosine annealing (on by default)')
parser.add_argument('--lr_decay_epochs', type=str, default='15,25',
                        help=' 15, 25, 40 for waterbibrds; 10, 15 ,20 for color_mnist')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
# saving, naming and logging
parser.add_argument('--exp_name', default = datetime.now().strftime("%m-%d-%Y_%H.%M.%S"), type=str, 
                    help='help identify checkpoint')
parser.add_argument('--project_name', default="testing", type=str,
                    help='name of experiment')
parser.add_argument('--log_name', type = str, default = "info.log",
                    help='Name of the Log File')
# Device options
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--local_rank', default=-1, type=int,
                        help='rank for the current node')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Reweighting
parser.add_argument('--reweight', type=int, default=0,
                            help='reweight the examples during training')
parser.add_argument('--joint_indep', type=int, default=0, help='add joint independence')
parser.add_argument('--_lambda', type=int, default=1,
                        help='lambda value for joint indep')
parser.add_argument('--marginal_indep', type=int, default=0,
                        help='add marginal independence, i.e. only r(x) \indep_p_\ind z')
parser.add_argument('--critic_restart', type=int, default=0)
parser.add_argument('--exact', type=int, default=0)
parser.add_argument('--reweight_val_only', type=int, default=0)
parser.add_argument('--undersample_val_only', type=int, default=0)
# wandb params
parser.add_argument('--local_testing', type=int, default=0,
                        help='add joint independence')
parser.add_argument('--reweight_model_eval', type=int, default=1,
                    help='turn reweight_model on eval')
# undersampling
parser.add_argument('--undersample', type=int, default=0)
# cxr data params
parser.add_argument('--in_disease', type=str, default="Pneumonia")
parser.add_argument('--in_disease_neg', type=str, default="No Finding")
parser.add_argument('--ood_disease', type=str, default="Lung Lesion")
parser.add_argument('--shuffled_label', type=int, default=0)
parser.add_argument('--random_init', type=int, default=0)
parser.add_argument('--fractal_db', type=int, default=0)
# celeba
parser.add_argument('--balance_correlation_sizes', type=int, default=0)
# cmnist
parser.add_argument('--label_bias', type=int, default=0)
parser.add_argument('--deterministic_label', type=int, default=0)
parser.add_argument('--model_file', default=None, help='path to saved model checkpoint')

args, unknown = parser.parse_known_args()
print(f"Unknown args in train.py: {unknown}")
if not args.local_testing:
    import wandb
    # run might exist from the outer main.py
    # if wandb.run is None:
    #     wandb.init(project=f'nurd-ood-{args.project_name}')
    #     wandb.config.update(args)
    wandb.init(id=args.exp_name, resume="allow", project="nurd-ood-" + args.project_name, reinit=True)
    wandb.config.update(args)


state = {k: v for k, v in args._get_kwargs()}

directory = "checkpoints/{in_dataset}/{name}/{exp}/".format(in_dataset=args.in_dataset, 
            name=args.project_name, exp=args.exp_name)
os.makedirs(directory, exist_ok=True)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

# CUDA Specification
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
set_random_seed(args.manualSeed)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def flatten(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

def get_weights(targets, nuisances, reweight_model, label_prior):
    _, y_pred_g_z = reweight_model(nuisances)
    inv_weights = (F.cross_entropy(y_pred_g_z, targets.long().view(-1), reduction='none').view(-1)*-1).exp()
    # print("p(y|z)", inv_weights.min(), inv_weights.max())
    prior_weights = torch.from_numpy(np.array([label_prior[t] for t in targets.cpu().data.numpy()])).to(device)
    # print("prior", prior_weights.min(), prior_weights.max())
    return prior_weights / inv_weights

def record_metrics(acc, loss, top1, inputs, outputs, targets, losses):
    prec1 = accuracy(outputs.data, targets, topk=(1,))[0]
    acc.update((torch.max(outputs, dim=1)[1].data == targets).sum().data / len(outputs), inputs.size(0))
    loss.update(losses.mean().data, inputs.size(0))
    top1.update(prec1, inputs.size(0))
    return acc, loss, top1

def record_rw_metrics(acc, loss, inputs, outputs, targets, losses, weights):
    num_correct = torch.max(outputs, dim=1)[1].data == targets
    acc.update((num_correct * weights).sum().data / weights.sum().data, inputs.size(0))
    loss.update((losses * weights).sum().data / weights.sum().data, inputs.size(0))
    return acc, loss

def log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss=None, rw_acc=None, split=None, log_wandb=True):
    log.debug('{split} Epoch: [{0}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
            epoch, batch_time=batch_time,
            loss=loss, top1=top1, acc=acc, split=split))
    if rw_loss and rw_acc:
        log.debug(
            'Rw Loss {rw_loss.val:.3f} ({rw_loss.avg:.3f})\t'
            'Rw Acc {rw_acc.val:.3f} ({rw_acc.avg:.3f})\t'.format(
                rw_loss=rw_loss, rw_acc=rw_acc))
    if log_wandb and not args.local_testing:
        wandb.log({
            f"{split} loss": loss.avg,
            f"{split} prec1": top1.avg,
            f"{split} acc": acc.avg}, step=epoch)
        if rw_loss and rw_acc:
            wandb.log({
                f"{split} rw_loss": rw_loss.avg,
                f"{split} rw_acc": rw_acc.avg}, step=epoch)


def train(model, train_loader, val_loader, criterion, optimizer, epoch, log, reweight_args={}, joint_indep_args={}, args=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    acc = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()
    rw_acc = AverageMeter()
    rw_loss = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    batch_idx = 0
    len_dataloader = 0
    len_dataloader += len(train_loader)
    for step, (inputs, targets, nuisances) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        nuisances = nuisances.to(device)
        if args.shuffled_label:
            targets = targets[torch.randperm(targets.size()[0])]
            
        
        if joint_indep_args["joint_indep"]:
            best_loss = None
            joint_indep_args["critic_model"] = unfreeze_model(joint_indep_args["critic_model"])
            model = freeze_model(model)
            critic_optimizer = torch.optim.Adam(joint_indep_args["critic_model"].parameters(), lr=joint_indep_args["lr"], weight_decay=joint_indep_args["weight_decay"])
            for critic_epoch in range(joint_indep_args["critic_epochs"]):
                joint_indep_args["critic_model"] = train_critic(joint_indep_args["critic_model"], model, train_loader, criterion, critic_optimizer, critic_epoch, log, 
                                                                reweight_args=reweight_args, joint_indep_args=joint_indep_args)
                critic_loss, critic_acc, critic_rw_acc = validate_critic(val_loader, joint_indep_args["critic_model"], model, criterion, critic_epoch, log,
                                                                        reweight_args=reweight_args, joint_indep_args=joint_indep_args)
                if not best_loss or critic_loss < best_loss:
                    best_loss = critic_loss
                    save_checkpoint(args, {
                            'epoch': critic_epoch + 1,
                            'state_dict_model': joint_indep_args["critic_model"].state_dict(),
                    }, epoch + 1, name="critic")
                    wandb.run.summary["best_critic_acc"] = critic_acc
                    wandb.run.summary["best_critic_rw_acc"] = critic_rw_acc
            # reload and freeze best critic model
            model_dir = "checkpoints/{in_dataset}/{name}/{exp}/".format(in_dataset=args.in_dataset, name=args.project_name, exp=args.exp_name)
            model_file = model_dir + 'checkpoint_critic.pth.tar'
            joint_indep_args["critic_model"].load_state_dict(torch.load(model_file)["state_dict_model"])
            joint_indep_args["critic_model"] = freeze_model(joint_indep_args["critic_model"])
            # unfreeze main model
            model = unfreeze_model(model)

        activations, outputs = model(inputs)
        
        losses = criterion(outputs, targets)

        # measure accuracy and record loss
        acc, loss, top1 = record_metrics(acc, loss, top1, inputs, outputs, targets, losses)

        # compute gradient and do SGD step
        if joint_indep_args["joint_indep"]:
            info_losses = estimate_mutual_information(inputs, targets, nuisances, model, joint_indep_args["critic_model"], criterion, reweight_args, joint_indep_args)
            losses = losses + joint_indep_args["lambda"] * info_losses

        if reweight_args["reweight"] and not reweight_args["reweight_val_only"]:
            weights = get_weights(targets, nuisances, reweight_args["reweight_model"], label_prior=reweight_args["label_prior"])
            # print("weights", weights.min(), weights.max())
            rw_acc, rw_loss = record_rw_metrics(rw_acc, rw_loss, inputs, outputs, targets, losses, weights)
            tensor_loss = (losses * weights).sum() / weights.sum()
        else:
            tensor_loss = losses.mean()

        optimizer.zero_grad()
        tensor_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
            
        batch_idx += 1
    log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss, rw_acc, split="Train")

def validate(val_loader, model, criterion, epoch, log, reweight_args={}, joint_indep_args={}):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    acc = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()
    rw_acc = AverageMeter()
    rw_loss = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for inputs, targets, nuisances in val_loader:
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            nuisances = nuisances.to(device)
            # compute output
            _, outputs = model(inputs)
            losses = criterion(outputs, targets)

            if joint_indep_args["joint_indep"]:
                info_losses = estimate_mutual_information(inputs, targets, nuisances, model, joint_indep_args["critic_model"], criterion, reweight_args, joint_indep_args)
                losses = losses + joint_indep_args["lambda"] * info_losses

            # measure accuracy and record loss
            acc, loss, top1 = record_metrics(acc, loss, top1, inputs, outputs, targets, losses)

            if reweight_args["reweight"]:
                weights = get_weights(targets, nuisances, reweight_args["reweight_model"], label_prior=reweight_args["label_prior"])
                # print("weights", weights.min(), weights.max())
                rw_acc, rw_loss = record_rw_metrics(rw_acc, rw_loss, inputs, outputs, targets, losses, weights)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss, rw_acc, split="Val")

    return_loss = rw_loss.avg if reweight_args["reweight"] else loss.avg
    return return_loss, acc.avg, rw_acc.avg

def train_reweight_model(model, train_loader, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    acc = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    len_dataloader = 0
    len_dataloader += len(train_loader)
    for _, targets, nuisances in train_loader:
        targets = targets.long().to(device)
        nuisances = nuisances.to(device)

        _, outputs = model(nuisances)
        
        losses = criterion(outputs, targets)

        # measure accuracy and record loss
        acc, loss, top1 = record_metrics(acc, loss, top1, nuisances, outputs, targets, losses)

        # compute gradient and do SGD step
        tensor_loss = losses.mean()

        optimizer.zero_grad()
        tensor_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    log_metrics(log, epoch, batch_time, loss, top1, acc, None, None, "Train Reweight")

def validate_reweight_model(val_loader, model, criterion, epoch, log):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    acc = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()

    len_dataloader = len(val_loader)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for _, targets, nuisances in val_loader:
            nuisances = nuisances.to(device)
            targets = targets.long().to(device)
            # compute output
            _, outputs = model(nuisances)
            losses = criterion(outputs, targets)

            # measure accuracy and record loss
            acc, loss, top1 = record_metrics(acc, loss, top1, nuisances, outputs, targets, losses)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    log_metrics(log, epoch, batch_time, loss, top1, acc, None, None, split="Val Reweight")

    return loss.avg, acc.avg

def estimate_mutual_information(inputs, labels, nuisances, model, critic_model, criterion, reweight_args, joint_indep_args):
    activations, _ = model(inputs)
    
    if joint_indep_args["marginal_indep"]:
        outputs = critic_model(activations, torch.zeros_like(labels.unsqueeze(1)).type(torch.FloatTensor).to(device), nuisances)
    else:
        outputs = critic_model(activations, labels.unsqueeze(1).type(torch.FloatTensor).to(device), nuisances)
    
    log_softmax_outputs = torch.log_softmax(outputs, dim=1)
    mi_terms = log_softmax_outputs[:, 1] - log_softmax_outputs[:, 0]
    return mi_terms

def compute_critic_loss(inputs, labels, nuisances, model, critic_model, criterion, reweight_args, joint_indep_args):
    activations, _ = model(inputs)
    
    if joint_indep_args["marginal_indep"]:
        positive_outputs = critic_model(activations, torch.zeros_like(labels.unsqueeze(1)).type(torch.FloatTensor).to(device), nuisances)
    else:
        positive_outputs = critic_model(activations, labels.unsqueeze(1).type(torch.FloatTensor).to(device), nuisances)
    pos_losses = criterion(positive_outputs, torch.ones_like(labels))

    shuffled_nuisances = nuisances[torch.randperm(nuisances.size()[0])]
    if joint_indep_args["marginal_indep"]:
        negative_outputs = critic_model(activations, torch.zeros_like(labels.unsqueeze(1)).type(torch.FloatTensor).to(device), shuffled_nuisances)
    else:
        negative_outputs = critic_model(activations, labels.unsqueeze(1).type(torch.FloatTensor).to(device), shuffled_nuisances)
    neg_losses = criterion(negative_outputs, torch.zeros_like(labels))

    outputs = torch.cat([positive_outputs, negative_outputs], dim=0)
    targets = torch.cat([torch.ones_like(labels), torch.zeros_like(labels)], dim=0)
    losses = torch.cat([pos_losses, neg_losses], dim=0)  # unweighted
    assert outputs.shape[0] == targets.shape[0] == 2 * len(inputs)
    return outputs, targets, losses

def train_critic(critic_model, model, train_loader, criterion, optimizer, epoch, log, reweight_args={}, joint_indep_args={}):
    batch_time = AverageMeter()
    acc = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()
    rw_acc = AverageMeter()
    rw_loss = AverageMeter()

    end = time.time()
    batch_idx = 0
    len_dataloader = 0
    len_dataloader += len(train_loader)
    for inputs, labels, nuisances in train_loader:
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        nuisances = nuisances.to(device)
        outputs, targets, losses = compute_critic_loss(
            inputs, labels, nuisances, model, critic_model, criterion, reweight_args, joint_indep_args)
        
        # unweighted metrics
        # measure accuracy and record loss
        acc, loss, top1 = record_metrics(acc, loss, top1, nuisances, outputs, targets, losses)

        # compute gradient and do SGD step
        if reweight_args["reweight"]:
            weights = get_weights(labels, nuisances, reweight_args["reweight_model"], label_prior=reweight_args["label_prior"])
        else:
            weights = torch.ones(len(labels)).to(device)
        # print("weights", weights.min(), weights.max())
        rw_acc, rw_loss = record_rw_metrics(rw_acc, rw_loss, inputs, outputs, targets, losses, weights.repeat(2))
        tensor_loss = (losses * weights.repeat(2)).sum() / weights.sum()

        optimizer.zero_grad()
        tensor_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        batch_idx += 1
    log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss, rw_acc, split="Train Critic", log_wandb=False)

    return critic_model

def validate_critic(val_loader, critic_model, model, criterion, epoch, log, reweight_args={}, joint_indep_args={}):
    batch_time = AverageMeter()
    acc = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()
    rw_acc = AverageMeter()
    rw_loss = AverageMeter()

    len_dataloader = len(val_loader)
    # switch to evaluate mode
    critic_model.eval()
    model.eval()
    with torch.no_grad():
        end = time.time()
        for inputs, labels, nuisances in val_loader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            nuisances = nuisances.to(device)
            outputs, targets, losses = compute_critic_loss(
                inputs, labels, nuisances, model, critic_model, criterion, reweight_args, joint_indep_args)
            
            # unweighted metrics
            # measure accuracy and record loss
            acc, loss, top1 = record_metrics(acc, loss, top1, nuisances, outputs, targets, losses)

            # compute gradient and do SGD step
            if reweight_args["reweight"]:
                weights = get_weights(labels, nuisances, reweight_args["reweight_model"], label_prior=reweight_args["label_prior"])
            else:
                weights = torch.ones(len(labels)).to(device)
            # print("weights", weights.min(), weights.max())
            rw_acc, rw_loss = record_rw_metrics(rw_acc, rw_loss, inputs, outputs, targets, losses, weights.repeat(2))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss, rw_acc, split="Val Critic", log_wandb=False)
    return_loss = rw_loss.avg if reweight_args["reweight"] else loss.avg
    return return_loss, acc.avg, rw_acc.avg


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    # leave learning rate constant otherwise
    # else:
    #     steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    #     if steps > 0:
    #         lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(directory, args.log_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 

    # get data
    if args.in_dataset == "waterbird":
        dataset_class = WaterbirdDataset
    elif args.in_dataset == "cmnist":
        dataset_class = ColoredMNIST
    else:
        raise ValueError(f"in_dataset not supported: {args.in_dataset}.")
    
    train_dataset = dataset_class(args, "train")
    val_dataset = dataset_class(args, "val")
    test_dataset = dataset_class(args, "test")

    kwargs = {'pin_memory': True, 'num_workers': 8, 'drop_last': True}
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    # label_prior = train_dataset.get_label_prior()
    label_prior = {0: .5, 1: .5}  # reweight to balance the labels
    
    balanced_test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    
    img_side_dict = {"waterbird": 224, "cxr": 224, "cxr-bal": 224, "cifar10": 32, "cmnist": 28, "cxr-small": 32, "celeba": 224}
    img_side = img_side_dict[args.in_dataset]
    num_classes_dict = {"waterbird": 2, "cxr": 2, "cxr-bal": 2, "cifar10": 10, "cmnist": 2, "cxr-small": 2, "celeba": 2}
    num_classes = num_classes_dict[args.in_dataset]

    if args.model_arch == "resnet18":
        base_model = load_model(orig=img_side==224, num_classes=num_classes)
    else:
        base_model = SimpleClassifier(img_side=img_side, num_classes=num_classes)
    
    if torch.cuda.device_count() > 1:
        base_model = torch.nn.DataParallel(base_model)


    model = base_model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if args.reweight:
        reweight_model = copy.deepcopy(base_model).to(device)
        reweight_optimizer = torch.optim.Adam(reweight_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        reweight_model = None
    
    if args.joint_indep:
        critic_model = CriticModel(img_side=img_side, num_classes=num_classes, model_arch=args.model_arch).to(device)
    else:
        critic_model = None
    
    reweight_args = {
        "reweight": args.reweight,
        "reweight_model": reweight_model,
        "label_prior": label_prior,
        "reweight_val_only": args.reweight_val_only,
    }
    
    joint_indep_args = {
        "joint_indep": args.joint_indep,
        "critic_model": critic_model,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "critic_epochs": args.critic_epochs,
        "marginal_indep": args.marginal_indep,
        "lambda": args._lambda,
    }

    cudnn.benchmark = True

    # keep batch norm logic simple: use training statistics during train
    # and running mean statistics during validation
    freeze_bn_affine = False
    def freeze_bn(model, freeze_bn_affine=True):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    freeze_bn(model, freeze_bn_affine)

    # train reweighted model
    best_loss = None
    if args.reweight:
        for epoch in range(0, args.reweight_epochs):
            log.debug(f"Start reweight model training epoch {epoch}")
            adjust_learning_rate(args, reweight_optimizer, epoch)
            train_reweight_model(reweight_model, train_loader, criterion, reweight_optimizer, epoch, log)
            loss, acc = validate_reweight_model(val_loader, reweight_model, criterion, epoch, log)
            if not best_loss or loss < best_loss:
                best_loss = loss
                log.debug("Saving reweight checkpoint")
                save_checkpoint(args, {
                        'epoch': epoch + 1,
                        'state_dict_model': reweight_model.state_dict(),
                }, epoch + 1, name="reweight")
                if not args.local_testing:
                    wandb.run.summary["best_val_reweight_acc"] = acc
    
    # reload and freeze best reweight_model
    if args.reweight:
        model_dir = "checkpoints/{in_dataset}/{name}/{exp}/".format(in_dataset=args.in_dataset, name=args.project_name, exp=args.exp_name)
        model_file = model_dir + 'checkpoint_reweight.pth.tar'
        reweight_model.load_state_dict(torch.load(model_file)["state_dict_model"])
        reweight_model = freeze_model(reweight_model)
        if args.reweight_model_eval:
            reweight_model.eval()
        reweight_args["reweight_model"] = reweight_model
        print("Reloaded best reweight_model")
    
    # train main model
    best_loss = None
    for epoch in range(args.start_epoch, args.epochs):
        log.debug(f"Start training epoch {epoch}")
        adjust_learning_rate(args, optimizer, epoch)
        train(model, train_loader, val_loader, criterion, optimizer, epoch + args.reweight_epochs, log,
            reweight_args=reweight_args, joint_indep_args=joint_indep_args, args=args)
        loss, acc, rw_acc = validate(val_loader, model, criterion, epoch + args.reweight_epochs, log, reweight_args, joint_indep_args)
        if not best_loss or loss < best_loss:
            best_loss = loss
            log.debug("Saving main checkpoint")
            save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
            }, epoch + 1, name="main") 
            if not args.local_testing:
                wandb.run.summary["best_val_acc"] = acc
                wandb.run.summary["best_val_rw_acc"] = rw_acc

    # evaluate on balanced test dataset
    model = get_model(args)
    loss, acc, rw_acc = validate(balanced_test_loader, model, criterion, args.epochs + args.reweight_epochs + 1, log, reweight_args, joint_indep_args)
    if not args.local_testing:
        wandb.run.summary["best_bal_test_acc"] = acc

if __name__ == '__main__':
    main()
    