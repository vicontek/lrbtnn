import logging
from functools import partial
from typing import *

from collections import OrderedDict
from typing import Sequence, Any, Iterable, Optional, List
import numpy as np
# import click
# import click_log
import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from find_mode import find_max_mode


class TTLayer(nn.Module):
    def __init__(self, in_factors, out_factors, ranks, ein_string, device='cpu'):
        super().__init__()
        self.in_factors = in_factors
        self.out_factors = out_factors
        self.ein_string = ein_string
        assert len(in_factors) == len(out_factors) == len(ranks) + 1, 'Input factorization should match output factorization and should be equal to len(ranks) - 1'
#         assert len(ranks) == 4, 'Now we consider particular factorization for given dataset'

        self.cores = nn.ParameterList([nn.Parameter(torch.randn(in_factors[0], 1, ranks[0], out_factors[0], ) * 0.8)])
        for i in range(1, len(in_factors) - 1):
            self.cores.append(nn.Parameter(torch.randn(in_factors[i], ranks[i-1], ranks[i], out_factors[i],) * 0.1))
        self.cores.append(nn.Parameter(torch.randn(in_factors[-1], ranks[-1], 1, out_factors[-1], ) * 0.8))
#         print(self.cores)
    def forward(self, x):
        reshaped_input = x.reshape(-1, *self.in_factors)
#         print('reshaped_input', reshaped_input.shape)
        # in the einsum below, n stands for index of sample in the batch,
        # abcde - indices corresponding to h1, h2, hw, w1, w2 modes
        # o, i, j, k, l, p - indices corresponding to the 4 tensor train ranks
        # v, w, x, y, z - indices corresponding to o1, o2, o3, o4, o5

        result = torch.einsum(
            self.ein_string,
            reshaped_input, *self.cores
        )
        return result.reshape(-1, np.prod(self.out_factors))
    
#     def parameters(self):
#         return self.cores

class TTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(OrderedDict([
             ('up', nn.Upsample(size=cfg.resize_shape, mode="bilinear", align_corners=False)),
            ('tt0', TTLayer(cfg.in_factors, cfg.hidd_out_factors, cfg.l1_ranks, cfg.ein_string1)),
            ('relu', nn.ReLU()),
#             nn.Linear(np.prod(hidd_factors), NUM_LABELS),
            ('tt1', TTLayer(cfg.hidd_in_factors, cfg.out_factors, cfg.l2_ranks, cfg.ein_string2)),
            # ('softmax', nn.Softmax(dim=1))
            ]),)

        # self.

    def forward(self, x):
        return self.net(x)
#     def parameters(self,):
#         return self.net[1].parameters() + list(self.net[3].parameters())


def get_lambdas_vec(lambdas) -> torch.Tensor:
    return torch.cat(tuple([rank for layer in lambdas for rank in layer]))


def vectorize_params(model, lambdas):
    params_vec = torch.cat(tuple(c.view(-1) for c in model.net.tt0.cores) + tuple(c.view(-1) for c in model.net.tt1.cores))
    return torch.cat((params_vec, get_lambdas_vec(lambdas)))


def calc_ranks_from_particles(
    cfg, particles: Sequence[torch.Tensor], kernel_sigma_squared: float,
    how: str = "mode", threshold: float = 1e-2
) -> Tuple[Tuple[int, int, int, int], Tuple[int]]:
    lambdas = tuple(
        unvectorize_params(particle, cfg)[1]
        for particle in particles
    )
    lambdas_vecs: np.ndarray = np.column_stack((
        get_lambdas_vec(lambda_).detach().cpu().numpy()
        for lambda_ in lambdas
    )).T
    lambda_mode = find_max_mode(
        lambdas_vecs, kernel_sigma_squared, sample_size=500000, verbose=True
    )
    cur = 0
    ranks = []
    assert lambda_mode.shape == (sum(cfg.l1_ranks) + sum(cfg.l2_ranks), )
    for layer_max_ranks in (cfg.l1_ranks, cfg.l2_ranks):
        this_layer_ranks = []
        for max_rank in layer_max_ranks:
            this_layer_ranks.append(np.sum(lambda_mode[cur:cur+max_rank] > threshold))
            cur += max_rank
        ranks.append(this_layer_ranks)
    return ranks
    
    
def unvectorize_params(theta, cfg):
    shapes = [np.array([4, 1, 8, 2]),
              np.array([4, 8, 8, 2]),
              np.array([4, 8, 8, 2]),
              np.array([4, 8, 8, 2]),
              np.array([4, 8, 1, 2]),
              np.array([4,  1, 16,  5]),
              np.array([8, 16,  1,  2])]
    curr = 0
    layer1_cores = []
    layer2_cores = []
    for i in range(len(shapes) - 2):
        layer1_cores.append(theta[curr: curr + np.prod(shapes[i])].view(*shapes[i]))
        curr += np.prod(shapes[i])
        
    for i in range(len(shapes) - 2, len(shapes)):
        layer2_cores.append(theta[curr: curr + np.prod(shapes[i])].view(*shapes[i]))
        curr += np.prod(shapes[i])
        
    state_dict = OrderedDict([])
    state_dict.update([('net.tt0.cores.' + str(i), w) for i, w in enumerate(layer1_cores)])
    state_dict.update([('net.tt1.cores.' + str(i), w) for i, w in enumerate(layer2_cores)])
    model = TTModel(cfg)
    model.load_state_dict(state_dict)
    
    lambdas = theta[curr:]
    lambdas = []
    for i, ranks in zip(range(2), [cfg.l1_ranks, cfg.l2_ranks]):
        lambdas.append([theta[curr + sum(ranks[0:i]): curr + sum(ranks[0:i+1])]
                                             for i in range(len(ranks))])
        curr += sum(ranks)
    return model, lambdas
