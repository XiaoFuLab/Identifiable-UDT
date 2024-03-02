import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing_extensions import Literal
from typing import Union


class Critic(nn.Module):
    def __init__(self, input_dim=2, hidden_units=128):
        super().__init__()
        self.main = nn.Sequential(
                        nn.Linear(input_dim, hidden_units),
                        # nn.BatchNorm1d(hidden_units),
                        nn.LayerNorm(hidden_units),
                        nn.LeakyReLU(0.2, True),

                        nn.Linear(hidden_units, hidden_units),
                        # nn.BatchNorm1d(hidden_units),
                        nn.LayerNorm(hidden_units),
                        nn.LeakyReLU(0.2, True),

                        nn.Linear(hidden_units, hidden_units),
                        # nn.BatchNorm1d(hidden_units),
                        nn.LayerNorm(hidden_units),
                        nn.LeakyReLU(0.2, True),

                        nn.Linear(hidden_units, hidden_units),
                        # nn.BatchNorm1d(hidden_units),
                        nn.LayerNorm(hidden_units),
                        nn.LeakyReLU(0.2, True),
                        
                        nn.Linear(hidden_units, 1)
                    )
        # self.grl = GradientReversalLayer()

    def forward(self, x):
        return self.main(x)
    
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, use_batchnorm=False, activation_func : Union[Literal['relu'], Literal['leaky_relu']]='leaky_relu'):
        super().__init__()
        activation_func_dict = {'relu': nn.ReLU,
                                'leaky_relu': nn.LeakyReLU}
        if activation_func not in activation_func_dict:
            raise ValueError(f'activation_func must be one of relu or leaky_relu, got {activation_func}')
        
        activation = activation_func_dict[activation_func]
        modules = [nn.Linear(input_dim, output_dim),
                    nn.BatchNorm1d(output_dim) if use_batchnorm else None,
                    activation(0.2, True) if activation_func=='leaky_relu' else activation(True)]
        modules = [m for m in modules if m is not None]
        self.main = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.main(x)

class Generator(nn.Module):
    def __init__(self, 
                input_dim=2,  
                output_dim=2,
                hidden_units=256,
                use_batchnorm=False,
                num_layers=2,
                activation_func : Union[Literal['relu'], Literal['leaky_relu']]='leaky_relu'):
        super().__init__()
        
        modules = [LinearBlock(input_dim, hidden_units, use_batchnorm, activation_func)]
        modules += [LinearBlock(hidden_units, hidden_units, use_batchnorm, activation_func) for _ in range(num_layers-1)]
        modules += [nn.Linear(hidden_units, output_dim)]
        modules = [item for item in modules if item is not None]

        self.pipe = nn.Sequential(*modules)

    def forward(self, x):
        return self.pipe(x)
