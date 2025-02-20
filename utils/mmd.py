import torch
from torch import nn


class RBFKernel(nn.Module):

    def __init__(self, scale: float = 1.0):
        super().__init__()

        self.scale = scale

    def forward(self, x, y):
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # Bx x 1
        y_norm = (y ** 2).sum(dim=1, keepdim=True)  # By x 1
        squared_ell_2 = x_norm - 2 * x @ y.T + y_norm.T  # Bx x By

        return torch.exp(-self.scale * squared_ell_2)


class BrayCurtisKernel(nn.Module):

    def __init__(self,):
        super().__init__()

    def forward(self, x, y):
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.abs(x - y).sum(dim=2)  # Bx x By
        denominator = torch.abs(x + y).sum(dim=2) + 1e-8

        return 1 - numerator / denominator


class TanimotoKernel(nn.Module):

    def __init__(self,):
        super().__init__()

    def forward(self, x, y):
        x = x.float()
        y = y.float()

        numerator = (x * y).sum(dim=1)  # Bx x By
        denominator = (x + y - x*y).sum(dim=1) + 1e-8

        return numerator / denominator


class RuzickaKernel(nn.Module):

    def __init__(self,):
        super().__init__()

    def forward(self, x, y):
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.min(x.unsqueeze(1), y.unsqueeze(0)).sum(dim=2)  # Bx x By
        denominator = torch.max(x.unsqueeze(1), y.unsqueeze(0)).sum(dim=2) + 1e-8

        return numerator / denominator


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBFKernel()):
        super().__init__()
        self.kernel = kernel

    def forward(self, x, y):
        k_xx = self.kernel(x, x)
        k_yy = self.kernel(y, y)
        k_xy = self.kernel(x, y)

        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()