#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    val_loader = data_loader.val_loader

    if args.update_bn:
        utils.update_bn(data_loader.train_loader, model, device=args.device)
    alpha = args.alpha1
    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            
            output, feats = model(data)
            # we're doing class 0 vs all
            output = torch.stack((output[:,0], output[:,1:].sum(axis=1)), dim=1)
            target = (~target.to(bool)).to(int)
            test_loss += criterion(output, target, torch.tensor([alpha, 1-alpha], device=args.device))

            # get the index of the max log-probability
            # might need to flip > if should go in the other direction
            pred = output.argmax(dim=1, keepdim=True)
            pred_eq = pred.eq(target.view_as(pred))
            correct += pred_eq.sum().item()
            # print(f'ALPHA1: {args.alpha1}\noutput_probs: {output_probs}\ntarget: {target}\npred: {pred}\npred_eq: {pred_eq}\ncorrect: {correct}')
            true_pos += pred[pred_eq].sum().item()
            false_pos += pred[~pred_eq].sum().item()
            true_neg += (1-pred)[pred_eq].sum().item()
            false_neg += (1-pred)[~pred_eq].sum().item()

    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader.dataset)
    test_sensitivity = true_pos / (true_pos + false_neg)
    test_specificity = true_neg / (true_neg + false_pos)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n" + 
        f'sensitivity: {test_sensitivity:.4f}, specificity: {test_specificity:.4f}\n'
    )

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/acc", test_acc, epoch)
        writer.add_scalar(f'test/sense', test_sensitivity, epoch)
        writer.add_scalar(f'test/spec', test_specificity, epoch)

    metrics = {
        'test_acc': test_acc,
        'test_specificity': test_specificity,
        'test_sensitivity': test_sensitivity
    }

    return test_acc, metrics
