import torch
import torchvision.models as models
import torch.nn as nn
from subspace_wrapper import to_subspace_class
import numpy as np


SubLinear = to_subspace_class(nn.Linear, num_vertices=3, verbose=True)

ls = SubLinear(5, 5)

ln = nn.Linear(5, 5)

incompatible_keys = ls.load_state_dict(ln.state_dict())

alpha = torch.full((num_vertices,), np.sqrt(num_vertices) / num_vertices)

print(ls.state_dict_at_alpha(alpha).keys())

SubResNet = to_subspace_class(models.ResNet, verbose=True)
sub_resnet18 = SubResNet(models.resnet.BasicBlock, [2, 2, 2, 2]).to(device)
resnet18 = models.resnet18().to(device)

incompatible_keys = sub_resnet18.load_state_dict(resnet18.state_dict())

print(incompatible_keys)
num_vertices = 2
alpha = torch.full((num_vertices,), np.sqrt(num_vertices) / num_vertices)
print(sub_resnet18.state_dict_at_alpha(alpha).keys())
