import subprocess
import argparse
import wandb
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dataset', default="celebA", type=str, help='in-distribution dataset e.g. IN-9')
    parser.add_argument('--data_label_correlation', default=0.95, type=float,
                        help='data_label_correlation')
    parser.add_argument('--model_arch', default='resnet18', type=str, help='model architecture e.g. resnet50')
    # ID train & val batch size
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        help='mini-batch size (default: 256) used for training')
    parser.add_argument('--eval_batch_size', default=10, type=int,
                        help='mini-batch size (default: 256) used for training')
    # training schedule
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of total epochs to run, default = 30')
    parser.add_argument('--reweight_epochs', default=0, type=int,
                        help='number of total epochs to run reweight model, default = 1')
    parser.add_argument('--critic_epochs', default=0, type=int,
                        help='number of total epochs to run reweight model, default = 1')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--optimizer', default="adam", type=str)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.005, type=float,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--exact', default=0, type=int,
                        help='use exact weights')
    parser.add_argument('--cosine', default=1, type=int, help='using cosine annealing (on by default)')
    # saving, naming and logging
    parser.add_argument('--project_name', default="erm_rebuttal", type=str,
                        help='name of experiment')
    parser.add_argument('--log_name', type = str, default = "info.log",
                        help='Name of the Log File')
    # Device options
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--local_rank', default=-1, type=int,
                            help='rank for the current node')
    # Miscs
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

    # Reweighting
    parser.add_argument('--reweight', type=int, default=0,
                            help='reweight the examples during training')
    parser.add_argument('--joint_indep', type=int, default=0,
                            help='add joint independence')
    parser.add_argument('--_lambda', type=int, default=1)
    parser.add_argument('--marginal_indep', type=int, default=0,
                            help='add marginal independence, i.e. only r(x) \indep_p_\ind z')                            
    parser.add_argument('--critic_restart', type=int, default=0)
    parser.add_argument('--reweight_model_eval', type=int, default=1,
                        help='turn reweight_model on eval')
    parser.add_argument('--undersample', type=int, default=0)
    parser.add_argument('--reweight_val_only', type=int, default=0)
    parser.add_argument('--undersample_val_only', type=int, default=0)
    # cxr
    parser.add_argument('--in_disease', type=str, default="Pneumonia")
    parser.add_argument('--in_disease_neg', type=str, default="No Finding")
    parser.add_argument('--ood_disease', type=str, default="Lung Lesion")
    # celeba
    parser.add_argument('--balance_correlation_sizes', type=int, default=0)
    # cmnist
    parser.add_argument('--label_bias', type=int, default=0)
    parser.add_argument('--shuffled_label', type=int, default=0)
    parser.add_argument('--deterministic_label', type=int, default=0)
    parser.add_argument('--model_file', default=None, type = str, help='path to saved model checkpoint')

    args = parser.parse_args()

    run = wandb.init(project="nurd-ood-" + args.project_name, settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args)

    args_list = []
    for i in vars(args):
        if "batch_size" not in i and "model_file" not in i:
            args_list.append("--"+i)
            args_list.append(str(getattr(args, i)))

    # exp_name = "_".join([method, "r" + args.data_label_correlation, "s" + args.maualSeed)
    exp_name = str(wandb.run.id)
    args_list.append("--exp_name")
    args_list.append(exp_name)

    result = subprocess.call(
        ["which", 'python']
        # capture_output=True, text=True
    )

    if args.exact:
        result = subprocess.call(
            ["python", "train_exact.py"] + args_list + ["--batch_size", str(args.batch_size)], 
            # capture_output=True, text=True
        )
    else:
        result = subprocess.call(
            ["python", "train.py"] + args_list + ["--batch_size", str(args.batch_size)], 
            # capture_output=True, text=True
        )
    # print("stdout:", result.stdout)
    # print("stderr:", result.stderr)

    torch.cuda.empty_cache()

    # add out dataset based on in dataset
    out_dataset = {
        "waterbird": "places365",
        "cxr": "cxr-other",
        "cxr-bal": "cxr-other",
        "cxr-small": "cxr-other-small",
        "cifar10": "cifar100",
        "cmnist": "cmnist-other",
        "celeba": "celeba-other"
    }
    args_list.append("--out_dataset")
    args_list.append(out_dataset[args.in_dataset])

    result = subprocess.call(
        ["python", "eval.py"] + args_list + ["--batch_size", str(args.eval_batch_size)], 
        # capture_output=True, text=True
    )
    # print("stdout:", result.stdout)
    # print("stderr:", result.stderr)