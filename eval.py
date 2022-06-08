# Derived from https://github.com/jfc43/robust-ood-detection/blob/master/CIFAR/eval_ood_detection.py
# Commit 7133fbaeb38efb64bb876e268a9008385aaa68c6
# Energy evaluation from https://github.com/deeplearning-wisc/Spurious_OOD/commit/4fe49c64ca9693004259e981d151ee0a550a3cdc


from __future__ import print_function
import argparse
import os

import multiprocessing
import sys
sys.path.append("..")

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import wandb
from scipy import misc
from utils import metric
from models.resnet import get_model
from utils.eval_utils import get_dataloaders, train_mahalanobis, process_mahalanobis
import logging
from utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in_dataset', required=True, type=str, help='in-distribution dataset')
parser.add_argument('--out_dataset', required=True, type=str, help='in-distribution dataset')
parser.add_argument('--data_label_correlation', required=True, type=float,
                    help='same as name in train.py')
parser.add_argument('--project_name', required=True, type=str,
                    help='same as name in train.py')
parser.add_argument('--exp_name', default=None, type=str,
                    help='same as exp in train.py')
parser.add_argument('--magnitude', default=0.0014, type=float,
                    help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int,
                    help='temperature scaling')
parser.add_argument('--model_arch', default='resnet18', type=str, help='model architecture e.g. resnet50')

parser.add_argument('--gpu', default = '0', type = str,
		    help='gpu index')
parser.add_argument('--adv', help='adv ood evaluation', action='store_true')

parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon')
parser.add_argument('--iters', default=10, type=int,
                    help='attack iterations')
parser.add_argument('--iter_size', default=1.0, type=float, help='attack step size')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch_size', default=10, type=int,
                    help='mini-batch size')
parser.add_argument('--local_testing', type=int, default=0)
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
# cxr data params
parser.add_argument('--in_disease', type=str, default="Pneumonia")
parser.add_argument('--in_disease_neg', type=str, default="No Finding")
parser.add_argument('--ood_disease', type=str, default="Lung Lesion")
parser.add_argument('--shuffled_label', type=int, default=0)
# other params
parser.add_argument('--undersample', type=int, default=0)
parser.add_argument('--undersample_val_only', type=int, default=0)
parser.add_argument('--exact', type=int, default=0)
parser.add_argument('--random_init', type=int, default=0)
parser.add_argument('--fractal_db', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=1)
# celeba
parser.add_argument('--balance_correlation_sizes', type=int, default=0)
parser.add_argument('--deterministic_label', type=int, default=0)

parser.add_argument('--model_file', default=None, help='path to saved model checkpoint')
parser.set_defaults(argument=True)

args, unknown = parser.parse_known_args()
print(f"Unknown args in eval.py: {unknown}")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

n_classes_dict = {
    "waterbird": 2, 
    "cxr": 2,
    "cifar10": 10,
    "cmnist": 2,
    "cxr-small": 2,
    "cxr-bal": 2,
    "celeba": 2
}

if not args.local_testing:
    if args.exp_name is not None:
        run = wandb.init(id=args.exp_name, project="nurd-ood-" + args.project_name, resume="allow", reinit=True)
        assert run is wandb.run
        exp_name = args.exp_name
    else:
        run = wandb.init(project="nurd-ood-" + args.project_name)
        exp_name = str(wandb.run.id)
    wandb.config.update(args, allow_val_change=True)


def MSP(outputs, model):
    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    return nnOutputs

def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    _, outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs

def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + args.in_dataset)
    print('out_distribution: '+ args.out_dataset)
    print('Model Name: ' + args.project_name)
    print('Under attack: ' + str(args.adv))
    print('')

    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
        print('')

def save_results(results, stypes, args, method):
    save_dir = os.path.join('output/ood_scores/', args.out_dataset, args.project_name, exp_name)
    fname = os.path.join(save_dir, f"results_{method}.txt")
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    with open(fname, "w") as f:
        f.write('in_distribution: ' + args.in_dataset + '\n')
        f.write('out_distribution: '+ args.out_dataset + '\n')
        f.write('Model Name: ' + args.project_name + '\n')
        f.write('Under attack: ' + str(args.adv) + '\n')
        f.write('\n')
    
        for stype in stypes:
            f.write(' OOD detection method: ' + stype + '\n')
            for mtype in mtypes:
                f.write(' {mtype:6s}'.format(mtype=mtype))
            f.write('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']))
            f.write(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']))
            f.write(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']))
            f.write(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']))
            f.write(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']))
            f.write('')
    
    if not args.local_testing and wandb.run is not None:
        for stype in stypes:
            wandb.run.summary[f"{stype}_fpr"] = 100.*results[stype]['FPR']
            wandb.run.summary[f"{stype}_dterr"] = 100.*results[stype]['DTERR']
            wandb.run.summary[f"{stype}_auroc"] = 100.*results[stype]['AUROC']
            wandb.run.summary[f"{stype}_auin"] = 100.*results[stype]['AUIN']
            wandb.run.summary[f"{stype}_auout"] = 100.*results[stype]['AUOUT']
    

def eval_mahalanobis():
    stypes = ['mahalanobis']

    save_dir = os.path.join('output/ood_scores/', args.out_dataset, args.project_name, exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start = time.time()

    trainloaderIn, testloaderIn, testloaderOut = get_dataloaders(args)

    model = get_model(args)

    model.eval()

    t0 = time.time()
    f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
    f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')

    ### train Mahalanobis detector
    print("Training Mahalanobis detector")
    sample_means, sample_cov = train_mahalanobis(trainloaderIn, model, n_classes_dict[args.in_dataset])

    ### evaluate in-distribution
    print("Processing in-distribution images")
    Mahalanobis_scores = process_mahalanobis(testloaderIn, model, sample_means, sample_cov)
    for k in range(len(testloaderIn)):
        f1.write("{}\n".format(-Mahalanobis_scores[k]))

    print("Processing out-of-distribution images")
    Mahalanobis_scores = process_mahalanobis(testloaderOut, model, sample_means, sample_cov)
    for k in range(len(testloaderOut)):
        f2.write("{}\n".format(-Mahalanobis_scores[k]))

    f1.close()
    f2.close()

    results = metric(save_dir, stypes)

    print_results(results, stypes)
    save_results(results, stypes, args, "mahalanobis")
    return

def eval_logits():
    stypes = ['LOGITS']

    save_dir = os.path.join('output/ood_scores/', args.out_dataset, args.project_name, exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start = time.time()
    #loading data sets

    _, testloaderIn, testloaderOut = get_dataloaders(args)

    model = get_model(args)
    model.eval()

    t0 = time.time()
    f1 = open(os.path.join(save_dir, "confidence_LOGITS_In.txt"), 'w')
    f2 = open(os.path.join(save_dir, "confidence_LOGITS_Out.txt"), 'w')
    print("Processing in-distribution images")
    count = 0
    t0 = time.time()
    for j, data in enumerate(testloaderIn):
        inputs, _, _ = data
        batch_size = inputs.shape[0]
        inputs = inputs.to(device)

        _, outputs = model(inputs)

        nnOutputs = outputs.data.cpu().numpy()

        for k in range(batch_size):
            f1.write("{}\n".format(np.max(nnOutputs[k])))

        count += 1
        t0 = time.time()


    t0 = time.time()
    print("Processing out-of-distribution images")
    count = 0

    for j, data in enumerate(testloaderOut):
        inputs, _, _ = data
        batch_size = inputs.shape[0]
        inputs = inputs.to(device)

        _, outputs = model(inputs)

        nnOutputs = outputs.data.cpu().numpy()

        for k in range(batch_size):
            f2.write("{}\n".format(np.max(nnOutputs[k])))

        count += 1
        t0 = time.time()


    f1.close()
    f2.close()

    results = metric(save_dir, stypes)

    print_results(results, stypes)
    save_results(results, stypes, args, "logits")


def eval_msp_and_odin():
    stypes = ['MSP', 'ODIN']

    save_dir = os.path.join('output/ood_scores/', args.out_dataset, args.project_name, exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start = time.time()
    #loading data sets

    _, testloaderIn, testloaderOut = get_dataloaders(args)

    model = get_model(args)
    model.eval()

    t0 = time.time()
    f1 = open(os.path.join(save_dir, "confidence_MSP_In.txt"), 'w')
    f2 = open(os.path.join(save_dir, "confidence_MSP_Out.txt"), 'w')
    g1 = open(os.path.join(save_dir, "confidence_ODIN_In.txt"), 'w')
    g2 = open(os.path.join(save_dir, "confidence_ODIN_Out.txt"), 'w')
    print("Processing in-distribution images")
    count = 0
    t0 = time.time()
    for j, data in enumerate(testloaderIn):
        inputs, _, _ = data
        batch_size = inputs.shape[0]
        inputs = inputs.to(device)
        inputs.requires_grad = True

        _, outputs = model(inputs)

        nnOutputs = MSP(outputs, model)

        for k in range(batch_size):
            f1.write("{}\n".format(np.max(nnOutputs[k])))

        nnOutputs = ODIN(inputs, outputs, model, temper=args.temperature, noiseMagnitude1=args.magnitude)

        for k in range(batch_size):
            g1.write("{}\n".format(np.max(nnOutputs[k])))

        count += 1
        t0 = time.time()


    t0 = time.time()
    print("Processing out-of-distribution images")
    count = 0

    for j, data in enumerate(testloaderOut):
        inputs, labels, _ = data
        batch_size = inputs.shape[0]
        inputs = inputs.to(device)
        inputs.requires_grad = True

        _, outputs = model(inputs)

        nnOutputs = MSP(outputs, model)

        for k in range(batch_size):
            f2.write("{}\n".format(np.max(nnOutputs[k])))

        nnOutputs = ODIN(inputs, outputs, model, temper=args.temperature, noiseMagnitude1=args.magnitude)

        for k in range(batch_size):
            g2.write("{}\n".format(np.max(nnOutputs[k])))

        count += 1
        t0 = time.time()


    f1.close()
    f2.close()
    g1.close()
    g2.close()

    results = metric(save_dir, stypes)

    print_results(results, stypes)
    save_results(results, stypes, args, "msp_and_odin")


def get_ood_energy(args, model, val_loader, epoch):
    in_energy = AverageMeter()
    model.eval()
    init = True
    print("######## Start collecting OOD energy score ########")
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(val_loader):
            images = images.to(device)
            _, outputs = model(images)
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy() 
            in_energy.update(e_s.mean(), len(labels))
            if init:
                sum_energy = e_s
                init = False
            else:
                sum_energy = np.concatenate((sum_energy, e_s))
            print('Epoch: [{0}] Batch#[{1}/{2}]\t'
                'OOD Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                    epoch, i, len(val_loader), in_energy=in_energy))
        return sum_energy

def get_id_energy(args, model, val_loader, epoch):
    in_energy = AverageMeter()
    top1 = AverageMeter()
    all_preds = torch.tensor([])
    all_targets = torch.tensor([])
    energy = np.empty(0)
    energy_grey = np.empty(0)
    energy_nongrey = np.empty(0)

    model.eval()
    print("######## Start collecting ID energy score ########")
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(val_loader):
            images = images.to(device)
            _, outputs = model(images)
            all_targets = torch.cat((all_targets, labels),dim=0)
            all_preds = torch.cat((all_preds, outputs.argmax(dim=1).cpu()),dim=0)
            prec1 = accuracy(outputs.cpu().data, labels, topk=(1,))[0]
            top1.update(prec1, images.size(0))
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy() 
            in_energy.update(e_s.mean(), len(labels)) 
            energy = np.concatenate((energy, e_s))
            energy_grey = np.concatenate((energy_grey, e_s[labels == 1]))
            energy_nongrey = np.concatenate((energy_nongrey, e_s[labels == 0]))
            print('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'ID Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                        epoch, i, len(val_loader), in_energy=in_energy))
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return energy, energy_grey, energy_nongrey


def eval_energy():

    # create model
    model = get_model(args)
    model = model.to(device)

    _, testloaderIn, testloaderOut = get_dataloaders(args)

    if args.in_dataset == 'color_mnist':
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.project_name, exp=exp_name)
    else:
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.project_name, exp=exp_name)
    
    # for test_epoch in test_epochs:
    test_epoch = 0
    if args.model_file is not None:
        cpts_path = args.model_file
        checkpoint = torch.load(cpts_path, map_location=device)
        state_dict = checkpoint['model_dict']
        rename_key = lambda k: k.replace("featurizer.network.", "").replace("classifier", "linear")
        state_dict = {rename_key(key): value for key, value in state_dict.items()
            if not key.startswith("network") and "num_batches_tracked" not in key}
        state_dict = {k: v for k, v in state_dict.items()
            if k.startswith("conv") or k.startswith("bn") or k.startswith("layer") or k.startswith("linear")}
    else:    
        cpts_path = os.path.join(cpts_directory, "checkpoint_main.pth.tar")
        checkpoint = torch.load(cpts_path, map_location=device)
        state_dict = checkpoint['state_dict_model']
    if torch.cuda.device_count() == 1:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    save_dir =  f"./energy_results/{args.in_dataset}/{args.project_name}/{args.exp_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("processing ID dataset")

    #********** normal procedure **********
    id_energy, _, _  = get_id_energy(args, model, testloaderIn, test_epoch)
    with open(os.path.join(save_dir, f'energy_score.npy'), 'wb') as f:
        np.save(f, id_energy)

    ood_energy = get_ood_energy(args, model, testloaderOut, test_epoch)
    with open(os.path.join(save_dir, f'energy_score_{args.out_dataset}.npy'), 'wb') as f:
        np.save(f, ood_energy)
    actual = np.concatenate((np.zeros_like(id_energy), np.ones_like(ood_energy)))
    preds = np.concatenate((id_energy, ood_energy))
    auc = roc_auc_score(actual, preds)
    if not args.local_testing:
        wandb.run.summary["energy_auroc"] = auc
    else:
        print(auc)

if __name__ == '__main__':
    eval_logits()        
    eval_msp_and_odin()
    eval_energy()
    eval_mahalanobis()
