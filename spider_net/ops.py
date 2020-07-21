import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable

# === OPERATION HELPERS ================================================================================================
def bracket(ops, ops2=None):
    out = ops + [nn.BatchNorm2d(ops[-1].out_channels, affine=True)]
    if ops2:
        out += [nn.ReLU(inplace=False)] + ops2 + [nn.BatchNorm2d(ops[-1].out_channels, affine=True)]
    return nn.Sequential(*out)

        
def dim_mod(dim, by_c, by_s):
    return dim[0], int(dim[1]*by_c), dim[2]//by_s, dim[3]//by_s


# from https://github.com/quark0/darts/blob/master/cnn/utils.py
def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def padsize(s=1, k=3, d=1):
    pad = math.ceil((k * d - d + 1 - s) / 2)
    return pad


# === GENERAL OPERATIONS ============================================================================================
class Zero(nn.Module):
    def __init__(self, stride, upscale=1):
        super().__init__()
        self.stride = stride
        self.upscale = upscale
        if stride == 1 and upscale == 1:
            self.op = lambda x: torch.zeros_like(x)
        else:
            self.op = lambda x: torch.zeros(dim_mod(x.shape, upscale, stride),
                                            device=torch.device('cuda'),
                                            dtype=x.dtype)

    def forward(self, x):
        return self.op(x)


class NNView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Sigmoid(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(x[:,:,::self.stride,::self.stride])   

class BatchNorm(nn.Module):
    def __init__(self, c, stride):
        super().__init__()
        self.stride = stride
        self.bn = nn.BatchNorm2d(c, affine=True)

    def forward(self, x):
        return self.bn(x[:,:,::self.stride,::self.stride])   
    
class ReLU(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x[:,:,::self.stride,::self.stride])
 

class MinimumIdentity(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.out_channels = c_out
        self.stride = stride
        if c_in == c_out:
            self.scaler = nn.Sequential()
        else:
            self.scaler = Scaler(c_in, c_out)

    def forward(self, x):
        x = self.scaler(x)
        if self.stride == 1:
            return x
        else:
            return x[:,:,::self.stride,::self.stride]



    
# === FFW OPERATIONS ================================================================================================
class ReluFC(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.op = nn.Sequential(
            nn.Linear(24,int(24*scale)),
            nn.ReLU(),
            nn.Linear(int(24*scale),24)
        )

    def forward(self, x):
        return self.op(x)
      
class SigmoidFC(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.op = nn.Sequential(
            nn.Linear(24,int(24*scale)),
            nn.Sigmoid(),
            nn.Linear(int(24*scale),24)
        )

    def forward(self, x):
        return self.op(x)  
    
# === CNN OPERATIONS ================================================================================================
class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    
    def forward(self,x):
        return self.op(x)

    
class SingleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        if c_in == c_out:
            self.op = bracket([
                nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            ])
        else:
            self.op = bracket([
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            ])

    def forward(self, x):
        return self.op(x)


class DilatedConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        self.op = bracket([
            nn.Conv2d(c_in, c_in, kernel_size, stride=stride, dilation=2, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, 1, padding=0, bias=False)
        ])

    def forward(self, x):
        return self.op(x)

    
class SeparableContConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()

        self.op = bracket([
            ContConv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ], [
            ContConv2d(c_in, c_in, kernel_size=kernel_size, stride=1, padding=padding, groups=c_in),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ] )

    def forward(self, x):
        return self.op(x)

class SeparableConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()

        self.op = bracket([
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ], [
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=1, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ] )

    def forward(self, x):
        return self.op(x)
    
class ISC(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()

        self.op = bracket([
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ], [
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=1, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ] )
        self.indentity = MinimumIdentity(c_in, c_out, stride)

    def forward(self, x):
        return self.indentity(x) + self.op(x)

class Scaler(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.op = nn.Conv2d(self.c_in, self.c_out, kernel_size=1, stride=1)

    def forward(self, x):
        return self.op(x)
        #return torch.cat([x]*self.c_out/self.c_in,dim=1)
    
        
class SELayer(nn.Module):
    def __init__(self, channel, stride, reduction=16):
        super().__init__()
        self.stride = stride
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        if self.stride>1:
            x = x[:,:,::self.stride,::self.stride]
        return  x * y.expand_as(x)

    
# === GAN OPERATIONS --===================================================================================================
class GeneratorTail(nn.Module):
    def __init__(self, position, preserve_aux, in_size, out_size):
        super().__init__()
        self.position = position
        self.preserve_aux = preserve_aux
        self.op = nn.Sequential(
            initializer(in_size, out_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.op(x)
  
    
# === UTITLITY OPERATIONS ================================================================================================
def padder(c_in, c_out, stride=1):
    return MinimumIdentity(c_in, c_out, stride=stride)


def initializer(c_in, c_out):
    return SingleConv(c_in, c_out, kernel_size=1, stride=1, padding=padsize(k=1, s=1))


def normalizer(c_in):
    return nn.BatchNorm2d(c_in, affine=True)


# === SEARCH SPACE =====================================================================================================
# commons = {
#     'Identity':     lambda c_in, s: MinimumIdentity(c_in, c_in, stride=s),
#     'Avg_Pool_3x3': lambda c_in, s: nn.AvgPool2d(3, stride=s, padding=padsize(s=s)),
#     'Max_Pool_3x3': lambda c_in, s: nn.MaxPool2d(3, stride=s, padding=padsize(s=s)),
#     'Max_Pool_5x5': lambda c_in, s: nn.MaxPool2d(5, stride=s, padding=padsize(k=5,s=s)),
#     'Max_Pool_7x7': lambda c_in, s: nn.MaxPool2d(7, stride=s, padding=padsize(k=7,s=s)),
#     'Conv_1x1':     lambda c_in, s: SingleConv(c_in, c_in, 1, stride=s, padding=padsize(k=1,s=s)),
#     'Conv_3x3':     lambda c_in, s: SingleConv(c_in, c_in, 3, stride=s, padding=padsize(k=3,s=s)),
#     'Conv_5x5':     lambda c_in, s: SingleConv(c_in, c_in, 5, stride=s, padding=padsize(k=5,s=s)),
#     'SE_8':         lambda c_in, s: SELayer(c_in, reduction=8, stride=s),
#     'SE_16':        lambda c_in, s: SELayer(c_in, reduction=16, stride=s),
#     'ReLU':         lambda c_in, s: ReLU(stride=s),
#     'Sigmoid':      lambda c_in, s: Sigmoid(stride=s),
#     'Sep_Conv_3x3': lambda c_in, s: SeparableConv(c_in, c_in, 3, stride=s, padding=padsize(s=s)),
#     'Sep_Conv_5x5': lambda c_in, s: SeparableConv(c_in, c_in, 5, stride=s, padding=padsize(k=5, s=s)),
#     'Sep_Conv_7x7': lambda c_in, s: SeparableConv(c_in, c_in, 7, stride=s, padding=padsize(k=7, s=s)),
#     'Dil_Conv_3x3': lambda c_in, s: DilatedConv(c_in, c_in, 3, stride=s, padding=padsize(d=2, s=s)),
#     'Dil_Conv_5x5': lambda c_in, s: DilatedConv(c_in, c_in, 5, stride=s, padding=padsize(d=2, k=5, s=s)),
# }



commons = {
    'Identity':     lambda c_in, s: MinimumIdentity(c_in, c_in, stride=s),
    'Max_Pool_3x3': lambda c_in, s: nn.MaxPool2d(3, stride=s, padding=padsize(s=s)),
    'Avg_Pool_3x3': lambda c_in, s: nn.AvgPool2d(3, stride=s, padding=padsize(s=s)),
    'Sep_Conv_3x3': lambda c_in, s: SeparableConv(c_in, c_in, 3, stride=s, padding=padsize(s=s)),
    'Sep_Conv_5x5': lambda c_in, s: SeparableConv(c_in, c_in, 5, stride=s, padding=padsize(k=5, s=s)),
    'Dil_Conv_3x3': lambda c_in, s: DilatedConv(c_in, c_in, 3, stride=s, padding=padsize(d=2, s=s)),
    'Dil_Conv_5x5': lambda c_in, s: DilatedConv(c_in, c_in, 5, stride=s, padding=padsize(d=2, k=5, s=s)),
}
