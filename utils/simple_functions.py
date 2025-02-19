import torch
import torch.nn.functional as F

EPS = 0.000001

def log1p(x):
    return torch.log(1. + x)

def log1p_inv(x):
    return torch.exp(x) - 1.

def log1p_inv_relu(x):
    return log1p_inv(torch.relu(x))

def id_fun(x):
    return x

def relu_fun(x):
    return torch.relu(x)

def round_relu_fun(x):
    return torch.round(torch.relu(x))

def normalize_fun(x):
    return x / x.sum(1, keepdim=True)

def softmax_fun(x):
    return torch.softmax(x, 1)

def tanh_fun(x):
    return torch.tanh(x)

def clip_11_fun(x):
    return torch.clip(x, -1., 1.)

def clip_fun(x, min_val: float=0., max_val: float=1.):
    assert min_val < max_val
    return torch.clip(x, min_val, max_val)

def log_pC(x, EPS: float=0.0001, C: float=1.0):
    return torch.log(x + EPS) + C
