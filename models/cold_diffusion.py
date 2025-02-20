from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning.pytorch as pl

from utils.mmd import MMDLoss, BrayCurtisKernel


class ColdDiffusionModel(pl.LightningModule):
    """Cold Diffusion Model"""

    def __init__(self, nnet, timesteps=1000, sigma_0=0.01, lr=1e-4, sequence_length=128, **kwargs):
        super().__init__()
        self.nnet = nnet  # DiT(sequence_length, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads)
        self.lr = lr
        # Define noise schedule
        self.timesteps = timesteps
        self.T_max = kwargs.get("T_max", 100)

        self.mmd = MMDLoss(kernel=BrayCurtisKernel())

        self.EPS = 1.e-6
        self.C = 2.

        self.mu_zero = 1.e-3
        self.sigma_min = 0.1
        self.sigma_max = 1.

    def forward(self, x, t):
        return self.nnet(x, t)

    def counts2simplex(self, x):
        total_counts = x.sum(-1, keepdim=True)
        y = x / (total_counts + self.EPS)
        return y, total_counts

    def simplex2real(self, x):
        y = torch.log(x + self.EPS) + self.C
        return y

    def counts2real(self, x):
        s, N = self.counts2simplex(x)
        y = self.simplex2real(s)
        return y, N

    def real2simplex(self, x):
        y = torch.softmax(x, -1)
        return y

    def simplex2counts(self, x, total_counts):
        y = torch.round(x * total_counts)
        return y

    def real2counts(self, x, total_counts):
        y = self.simplex2counts(self.real2simplex(x) + self.EPS, total_counts)
        return y

    def degradation(self, x_0, t):
        # Counts -> simplex -> real
        y_0, N = self.counts2real(x_0)

        # Real + noise
        mu_t = ((1. - t) * y_0) # t=1 -> mu_1 = 0, t=0 -> mu_0 = y_0

        sigma_t = ((1. - t) * self.sigma_min) + (t * self.sigma_max)

        y_t = mu_t * sigma_t * torch.randn_like(x_0)

        # Real + noise -> simplex -> counts
        x_t = self.real2counts(y_t, N)

        return x_t

    def restoration(self, x_t, t):
        # Counts -> simplex -> real
        y_t, N = self.counts2real(x_t)

        y_tilde_0 = self.nnet(y_t, t)

        # Real -> simplex -> counts
        x_tilde_0 = self.real2counts(y_tilde_0, N)

        return x_tilde_0

    @torch.no_grad()
    def sample(self, shape, total_counts=None):
        # t = 1
        # sample total counts
        if total_counts is None:
            total_counts = torch.randint(low=1400, high=1800, size=shape, device=self.device)
        # sample real
        y = torch.randn(shape, device=self.device)
        # get counts
        x = self.real2counts(y, total_counts)

        t_delta = 1 / self.timesteps

        for t in torch.linspace(1.-t_delta, t_delta, steps=self.timesteps-1):
            # Convert timestep into batch tensor
            t_tensor = torch.full((shape[0], 1), t, dtype=torch.float32, device=self.device)

            # Compute model prediction
            x_tilde = self.restoration(x, t_tensor)

            x = x - self.degradation(x_tilde, t_tensor) + self.degradation(x_tilde, t_tensor - t_delta)

        return x

    def loss_fun(self, x, t):
        x_t = self.degradation(x, t)
        x_tilde = self.restoration(x_t, t)

        loss = self.mmd(x, x_tilde)
        # loss = F.mse_loss(x, x_tilde)

        return loss

    def loss_fun_full(self, x, batch_idx):
        batch_sample = self.sample(shape=(x.shape[0], x.shape[1]))
        loss = self.mmd(x, batch_sample)

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]
        t = torch.rand(size=(1,1), device=self.device).repeat((batch_size, 1))
        loss = self.loss_fun(x, t)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        # loss = self.loss_fun_full(x, batch_idx)
        batch_size = x.shape[0]
        t = torch.rand(size=(1, 1), device=self.device).repeat((batch_size, 1))
        loss = self.loss_fun(x, t)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        # loss = self.loss_fun_full(x, batch_idx)
        batch_size = x.shape[0]
        t = torch.rand(size=(1, 1), device=self.device).repeat((batch_size, 1))
        loss = self.loss_fun(x, t)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer with Cosine Annealing Scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99))
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.lr * 0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}