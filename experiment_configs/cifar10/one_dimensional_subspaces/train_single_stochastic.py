#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import sys
import torch.nn.functional as F

sys.path.append(os.path.abspath("."))


from args import args
from main import main as run

if __name__ == "__main__":

    for seed in range(2):
        args.seed = seed
        args.label_noise = 0.0
        args.beta = 0
        args.layerwise = False
        args.num_samples = 1

        args.test_freq = 10
        args.set = "CIFAR10"
        args.multigpu = [1]
        args.model = "CIFARResNet"
        args.model_name = "cifar_resnet_20"
        args.conv_type = "StandardConv"
        args.bn_type = "StandardBN"
        args.conv_init = "kaiming_normal"
        args.trainer = "train_single_one_v_all_stochastic"
        args.warmup_length = 5
        args.data_seed = 0
        args.train_update_bn = True
        args.update_bn = True
        args.criterion = F.cross_entropy
        
        for epochs in [160, 160 * 2]:
            for optimizer in ['sgd']:
                for lr in [0.1]:
                    args.epochs = epochs
                    args.lr = lr
                    args.optimizer = optimizer
                    args.name = (
                        f"id=default_stochastic+ln={args.label_noise}"
                        f"+num_samples={args.num_samples}"
                        f"+optimizer={args.optimizer}"
                        f"+lr={args.lr}"
                        f"+epochs={args.epochs}"
                        f"+seed={args.seed}"
                    )

                    args.save = True
                    args.save_epochs = []
                    args.save_iters = []

                    # TODO: change these paths -- this is an example.
                    args.data = "/lfs/local/0/nomir/learning-subspaces/data"
                    args.log_dir = (
                        "/lfs/local/0/nomir/learning-subspaces/learning-subspaces-results/cifar/one-dimensional-subspaces"
                    )

                    run()

