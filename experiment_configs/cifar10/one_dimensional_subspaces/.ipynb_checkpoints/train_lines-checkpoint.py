#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import sys

sys.path.append(os.path.abspath("."))


from args import args
from main import main as run

if __name__ == "__main__":

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
        args.conv_type = "LinesConv"
        args.bn_type = "LinesBN"
        args.conv_init = "kaiming_normal"
        args.trainer = "train_one_dim_subspaces"
        args.epochs = 160
        args.warmup_length = 5
        args.data_seed = 0
        args.train_update_bn = True
        args.update_bn = True

        args.name = (
            f"id=lines+ln={args.label_noise}"
            f"+beta={args.beta}"
            f"+num_samples={args.num_samples}"
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
