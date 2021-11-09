import torch
from typing import Type, Optional, Union
import torch.nn as nn

def to_subspace_class(model_class: Type[nn.Module], num_vertices: Optional[int] = 2, verbose: Optional[bool] = False):
    class subspace_model_class(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.verbose = verbose
            # These will be the vertices of our n-simplex
            self.register_buffer('num_vertices', torch.Tensor([num_vertices]))
            # Have to register all of these as parameters (not registers) so that they're copied to the correct device
            # and so that they get backpropagated automatically
            self.parametrization_points = [nn.ParameterList([nn.Parameter(p.clone()) for _ in range(num_vertices)]) for p in self.parameters()]
            self.register_buffer('alpha', torch.ones(num_vertices) / num_vertices)

        def set_alpha(self, alpha: torch.Tensor):
            if alpha.size[0] != self.num_vertices:
                raise ValueError(f'Alpha must have size of at least self.num_vertices={self.num_vertices}')
            self.alpha = alpha

        # def to(self, device: Union[str, torch.device]):
        #     super().to(device)
        #     if verbose: 
        #         print(f'Sending custom parametrization points to {device}')
        #     for i in range(len(self.parametrization_points)):
        #         for j in range(int(self.num_vertices)):
        #             self.parametrization_points[i][j] = self.parametrization_points[i][j].to(device)

        def forward(self, x):
            assert self.alpha is not None, 'Alpha must be defined before a forward passs. Call model.set_alpha(<alpha>).'
            # set the parameters according to alpha
            if self.verbose:
                print('Setting parameters from subspace parametrization...')
            for i, p in enumerate(self.parameters()):
                to_stack = [self.parametrization_points[i][j] * self.alpha[j] for j in range(int(self.num_vertices))]
                p = torch.mean(torch.stack(to_stack, dim=0), axis=0)
            if self.verbose:
                print('Done setting parameters!')
            return super().forward(x)

    return subspace_model_class
