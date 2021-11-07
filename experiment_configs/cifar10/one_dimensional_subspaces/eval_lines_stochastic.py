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

    # TODO: change these paths -- this is an example.
    args.data = "/lfs/local/0/nomir/learning-subspaces/data"
    args.log_dir = (
        "/lfs/local/0/nomir/learning-subspaces/learning-subspaces-results/cifar/eval-one-dimensional-subspaces"
    )

    for seed in range(2):
        args.seed = seed
        args.lr = 0.1
        args.label_noise = 0.0
        args.beta = 0
        args.layerwise = False
        args.num_samples = 1

        args.test_freq = 10
        args.set = "CIFAR10"
        args.multigpu = [9]
        args.model = "CIFARResNet"
        args.model_name = "cifar_resnet_20"
        args.conv_type = "LinesConv"
        args.bn_type = "LinesBN"
        args.conv_init = "kaiming_normal"
        for optimizer in ['sgd']:
            for lr in [0.2]:
                for epochs in [160]:
                    args.epochs = epochs
                    args.lr = lr
                    args.optimizer = optimizer
                    name_string = (
                        f"id=lines_stochastic+ln={args.label_noise}"
                        f"+num_samples={args.num_samples}"
                        f"+optimizer={args.optimizer}"
                        f"+lr={args.lr}"
                        f"+epochs={args.epochs}"
                        f"+seed={args.seed}"
                    )

                    # Now, analyze.

                    args.resume = [
                        f"/lfs/local/0/nomir/learning-subspaces/learning-subspaces-results/cifar/one-dimensional-subspaces/{name_string}+try=0/"
                        f"epoch_{args.epochs}_iter_{args.epochs * round(50000 / 128)}.pt"
                    ] * 2

                    args.num_models = 2
                    args.save = False
                    args.save_data = True
                    args.pretrained = True
                    args.epochs = 0
                    args.trainer = "eval_one_dim_subspaces_stochastic"
                    args.update_bn = True

                    # Need to use weighted cross entropy
                    args.criterion = F.cross_entropy

                    acc_data = {}
                    for i, alpha0 in enumerate(
                        [
                            0.0,
                            0.05,
                            0.1,
                            0.15,
                            0.2,
                            0.25,
                            0.3,
                            0.35,
                            0.4,
                            0.45,
                            0.5,
                            0.5,
                            0.55,
                            0.6,
                            0.65,
                            0.7,
                            0.75,
                            0.8,
                            0.85,
                            0.9,
                            0.95,
                            1.0,
                        ]
                    ):
                        args.alpha0 = alpha0
                        args.alpha1 = 1.0 - alpha0
                        args.name = (
                            f"{name_string}+alpha0={args.alpha0}+alpha1={args.alpha1}"
                        )
                        args.save_epochs = []
                        run()
