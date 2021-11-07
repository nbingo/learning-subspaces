#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import sys
sys.path.append(os.path.abspath("."))
import torch.nn.functional as F
from tqdm import tqdm


from args import args
from main import main as run

if __name__ == "__main__":

    # TODO: change these paths -- this is an example.
    args.data = "/lfs/local/0/nomir/learning-subspaces/data"
    args.log_dir = (
        "/lfs/local/0/nomir/learning-subspaces/learning-subspaces-results/cifar/one-dimensional-subspaces"
    )

    for seed in range(2):
        args.seed = seed
        args.lr = 0.1
        args.label_noise = 0.0
        args.beta = 1.0
        args.layerwise = False
        args.num_samples = 1

        args.test_freq = 10
        args.set = "CIFAR10"
        args.multigpu = [8]
        args.model = "CIFARResNet"
        args.model_name = "cifar_resnet_20"
        args.conv_type = "StandardConv"
        args.bn_type = "StandardBN"
        args.conv_init = "kaiming_normal"
        args.trainer = "train_weighted_single_one_v_all"
        args.warmup_length = 5
        args.data_seed = 0
        args.train_update_bn = True
        args.update_bn = True

        args.criterion = F.cross_entropy
        for epochs in [160]:
            args.epochs = epochs
            name_string = (
                f"id=weighted_default_warm+ln={args.label_noise}"
                f"+num_samples={args.num_samples}"
                f"+optimizer={args.optimizer}"
                f"+lr={args.lr}"
                f"+epochs={args.epochs}"
                f"+seed={args.seed}"
            )

            args.save = True
            args.save_epochs = []
            args.save_iters = []

            acc_data = {}
            for i, alpha0 in tqdm(enumerate(range(35,70,5))):
                args.alpha0 = alpha0 / 100
                args.alpha1 = 1.0 - alpha0
                args.name = (
                    f"{name_string}+alpha0={args.alpha0}+alpha1={args.alpha1}"
                )
                args.save_epochs = []
                
                resume_name_string = (
                        f"id=default+ln={args.label_noise}"
                        f"+num_samples={args.num_samples}"
                        f"+optimizer={args.optimizer}"
                        f"+lr={args.lr}"
                        f"+epochs={args.epochs}"
                        f"+seed={args.seed}"
                )
                args.pretrained = True
                args.resume = [
                    f"{args.log_dir}/{resume_name_string}+try=0/"
                    f"epoch_{args.epochs}_iter_{args.epochs * round(50000 / 128)}.pt"
                ]
                    
                run()
