#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
from torch_scatter import scatter

import utils
from args import args


def init(models, writer, data_loader):
    return


def train(models, writer, data_loader, optimizers, criterion, epoch):
    return


def test(models, writer, criterion, data_loader, epoch):

    model = models[0]

    model.apply(lambda m: setattr(m, "return_feats", True))

    model.zero_grad()
    model.eval()
    test_loss = 0
    # numer correct in total
    correct = 0
    # number correct per-class
    if args.set == 'CIFAR10':
            num_classes = 10
    else:
            raise RuntimeError('Unknown number of classes')
    # number correct per-class
    class_correct = torch.zeros(num_classes, device=args.device, dtype=float)
    val_loader = data_loader.val_loader

    model.apply(lambda m: setattr(m, "alpha", args.alpha1))

    if args.update_bn:
        utils.update_bn(data_loader.train_loader, model, device=args.device)

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)

            output, feats = model(data)
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            pred_eq = pred.eq(target.view_as(pred)).to(float)
            #total correct
            correct += pred_eq.sum().item()
            # number correct per class
            # print(f'pred_eq shape: {pred_eq.shape}\ntarget shape: {target.view_as(pred).shape}\nclass_correct shape: {class_correct.shape}\nScatter shape: {scatter(pred_eq, target.view_as(pred)).shape}')
            # print(f'pred_eq dtype: {pred_eq.dtype}\ntarget dtype: {target.dtype}')
            class_correct = scatter(pred_eq.squeeze(), target.view_as(pred).squeeze(), out=class_correct)

    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader.dataset)
    class_test_acc = class_correct / (len(val_loader.dataset) / num_classes) #assuming equal number of examples per class
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n" + 
        f'Test set: Accuracy per-class {class_test_acc}\n'
    )

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/acc", test_acc, epoch)

    metrics = {
        'test_acc': test_acc,
        'class_acc': class_test_acc
    }

    return test_acc, metrics
