import torch
import torch.nn.functional as F

EPS = 0.000001
CONST = 1.

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

# data transforms
def counts2simplex(x):
    total_counts = x.sum(-1, keepdim=True)
    y = x / (total_counts + EPS)
    return y, total_counts

def simplex2real(x):
    y = torch.log(x + EPS) + CONST
    return y

def counts2real(x):
    s, N = counts2simplex(x)
    y = simplex2real(s)
    return y, N

def real2simplex(x):
    y = torch.softmax(x, -1)
    return y

def simplex2counts(x, total_counts):
    y = torch.round(x * total_counts)
    return y

def real2counts(x, total_counts):
    y = simplex2counts(real2simplex(x) + EPS, total_counts)
    return y