import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable

from spider_net.ops import *
from spider_net.helpers import width_mod, general_num_params


# === BASE PRUNER =================================================================================
class Pruner(nn.Module):
    def __init__(self, m=1e9, mem_size=0, init=None):
        super().__init__()
        if init is None:
            init = .01
        elif init == 'off':
            init = -1.
        elif type(init) is int:
            init = float(init)
        self.init = init
        self.mem_size = mem_size
        self.weight = nn.Parameter(torch.tensor([init]))
        self.m = m
        self.m_inv = 1/m
        self.weight_history = [0]

    def __str__(self):
        return 'Pruner'

    def reset_parameters(self):
        self.weight = nn.Parameter(torch.tensor([self.init]))
        self.weight_history = [0]

    def num_params(self):
        # return number of differential parameters of input model
        return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    def track_gates(self):
        self.weight_history[-1] += self.gate().item()


    def get_deadhead(self, prune_epochs, prune_interval, verbose=False, force=False):
        if len(self.weight_history) < prune_epochs and not force:
            self.weight_history.append(0)
            return False
        deadhead = (prune_interval * .25) > sum(self.weight_history[-prune_epochs:]) or force
        if deadhead:
            self.switch_off()
        if verbose:
            print(self.weight_history, deadhead)
        self.weight_history = self.weight_history[-prune_epochs:]
        self.weight_history.append(0)
        return deadhead

    def switch_off(self):
        for param in self.parameters():
            param.requires_grad = False

    def clamp(self):
        pre = self.weight.item()
        bound = self.init * 5
        if self.weight > bound:
            self.weight.data = self.weight.data * bound/self.weight.data
        elif self.weight < -bound:
            self.weight.data = self.weight.data * -bound/self.weight.data
            #2print(pre, self.weight.item())
         
    def saw(self):
        return torch.remainder(self.weight, self.m_inv)
        
    def gate(self):
        return self.weight > 0

    def sg(self):
        return self.saw() + self.gate()

    def forward(self, x):
        return self.sg() * x


# === OP + PRUNER COMBO =================================================================================
class PrunableOperation(nn.Module):
    def __init__(self, op_function, name, mem_size, c_in, stride, start_idx=0, pruner_init=None, prune=True):
        super().__init__()
        self.op_function = op_function
        self.stride = stride
        self.name = name
        self.orig_name = name
        self.op = self.op_function(c_in, stride)
        self.zero = name == 'Zero'
        self.prune = prune
        if self.prune:
            self.pruner = Pruner(mem_size=mem_size, init=pruner_init)
        if pruner_init == 'off':
            self.zero = True
            self.pruner.switch_off()
        self.analytics = {'pruner': [None]*start_idx,
                          'pruner_sg': [None] * start_idx,
                          'grad': [None]*start_idx}

    def track_gates(self):
        if not self.zero:
            self.pruner.track_gates()

    def get_growth_factor(self):
        return {'weight': self.pruner.weight.item(),
                'grad': self.pruner.weight.grad.item() if self.pruner.weight.grad is not None else None}

    def deadhead(self, prune_epochs, prune_interval, force=False):
        if self.zero or not self.pruner.get_deadhead(prune_epochs, prune_interval, force=force):
            return 0
        else:
            self.op = Zero(self.stride)
            self.name = "Zero"
            self.zero = True
            return 1

    def log_analytics(self):
        if not self.zero:
            weight = self.pruner.weight
            self.analytics['pruner'].append(weight.item())
            self.analytics['pruner_sg'].append(self.pruner.sg().item())
            self.analytics['grad'].append(weight.grad.item() if weight.grad else None)
        else:
            self.analytics['pruner'].append(None)
            self.analytics['pruner_sg'].append(None)
            self.analytics['grad'].append(None)

    def __str__(self):
        return self.name

    def forward(self, x):
        if self.prune:
            return self.op(x) if self.zero else self.pruner(self.op(x))
        else:
            return self.op(x)

class PrunableTower(nn.Module):
    def __init__(self, position, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.position = position
        self.analytics = {'pruner': [],
                          'pruner_sg': [],
                          'grad': []}
        self.pruner = Pruner()

        self.ops = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            NNView(),
            nn.Linear(self.in_size[1], self.out_size)
        )

    def track_gates(self):
        weight = self.pruner.weight
        self.analytics['pruner'].append(weight.item())
        self.analytics['pruner_sg'].append(self.pruner.sg().item())
        self.analytics['grad'].append(weight.grad.item() if weight.grad else None)

    def forward(self, x):
        return self.ops(x)

# === INPUT HANDLER FOR PRUNED CELL INPUTS
class PrunableInputs(nn.Module):
    def __init__(self, dims, scale_mod, genotype, random_ops, prune=True):
        super().__init__()
        # weight inits
        if genotype.get('weights') is None:
            pruner_inits = [None]*len(dims)
        else:
            pruner_inits = genotype.get('weights')

        # zero inits
        if genotype.get('zeros') is None:
            self.zeros = [False] * len(dims)
        else:
            self.zeros = genotype.get('zeros')

        if random_ops is not None:
            self.zeros = [False if np.random.rand()<(random_ops['i_c']/len(self.zeros)) else True for i in self.zeros]
            if all(self.zeros):
                self.zeros[np.random.choice(range(len(self.zeros)))]=False
        self.unified_dim = dims[-1]
        self.prune = prune
        ops, strides, upscales, pruners = [], [], [], []

        for i, dim in enumerate(dims):
            stride = self.unified_dim[1]//dim[1] if dim[3] != self.unified_dim[3] else 1
            strides.append(stride)
            c_in, c_out = dim[1], self.unified_dim[1]
            upscales.append(c_out/c_in)
            if self.zeros[i]:
                ops.append(Zero(stride, c_out/c_in))
            else:
                ops.append(MinimumIdentity(c_in, c_out, stride))
            if self.prune:
                pruners.append(Pruner(init=pruner_inits[i]))

        self.ops = nn.ModuleList(ops)
        self.strides = strides
        self.upscales = upscales
        if self.prune:
            self.pruners = nn.ModuleList(pruners)
        self.scaler = MinimumIdentity(self.unified_dim[1], self.unified_dim[1]*scale_mod, stride=1)

    def track_gates(self):
        [pruner.track_gates() for pruner in self.pruners]

    def deadhead(self, prune_interval):
        out = 0
        for i, pruner in enumerate(self.pruners):
            if self.zeros[i] or not pruner.get_deadhead(prune_interval):
                out += 0
            else:
                self.ops[i] = Zero(self.strides[i], self.upscales[i])
                self.zeros[i] = True
                out += 1
        return out

    def get_ins(self):
        return [i - 1 if i else 'In' for i, zero in enumerate(self.zeros) if not zero]

    def __str__(self):
        return str(self.get_ins())

    def forward(self, xs):
        if self.prune:
            out = sum([op(xs[i]) if self.zeros[i] else self.pruners[i](op(xs[i])) for i,op in enumerate(self.ops)])
        else:
            out = sum([op(xs[i]) for i, op in enumerate(self.ops)])
        return self.scaler(out)