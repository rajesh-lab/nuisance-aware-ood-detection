import os
import glob
import wandb
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--sweep_name', required=True, help="/lily/nurd-ood-repro_ming/sweeps/8ty1c9bx", type=str)
    args = parser.parse_args()
    api = wandb.Api()
    print(args.sweep_name)
    sweep = api.sweep(f"{args.sweep_name}")
    wandb_dir = "/scratch/lhz209/nood/nuisance_ood/src/supervised/wandb/"
    for run in sweep.runs:
        # print(run.id)
        run_dir_list = glob.glob(os.path.join(wandb_dir, f"run*{run.id}"))
        # take latest dir
        # run_dir = sorted(run_dir_list)[-1]
        # take the earliest dir
        run_dir = sorted(run_dir_list)[0]
        print(run_dir)
        # result = subprocess.call(["wandb", "sync", run_dir])
    
    # wandb_dir = "/scratch/lhz209/nood/nuisance_ood/src/supervised/wandb/"
    # for run_dir in glob.glob(os.path.join(wandb_dir, f"run-20220314*")):
    #     result = subprocess.call(["wandb", "sync", run_dir])