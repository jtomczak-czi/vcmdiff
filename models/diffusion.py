from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning.pytorch as pl

from utils.mmd import MMDLoss, BrayCurtisKernel


class DiffusionModel(pl.LightningModule):
    """Diffusion Model"""

    def __init__(self, nnet, timesteps=1000, beta_min=1e-4, beta_max=0.02, lr=1e-4, sequence_length=128, **kwargs):
        super().__init__()
        self.nnet = nnet  # DiT(sequence_length, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads)
        self.lr = lr
        # Define noise schedule
        self.timesteps = timesteps
        betas = torch.linspace(beta_min, beta_max, timesteps, device=self.device)
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.T_max = kwargs.get("T_max", 100)

    def forward(self, x, t):
        return self.nnet(x, t)

    def q_sample(self, x_start, t):
        """Adds Gaussian noise to the input signal according to the diffusion.py schedule."""
        noise = torch.randn_like(x_start)
        self.alpha_cumprod = self.alpha_cumprod.to(x_start.device)
        alpha_bar_t = self.alpha_cumprod[t].view(-1, 1, 1).to(x_start.device)
        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise, noise

    def simple_loss(self, x, t):
        x_noisy, noise = self.q_sample(x, t)
        predicted_noise = self(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def simple_loss_full(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]
        loss = 0.

        for t_iter in range(self.timesteps):
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device) * 0 + t_iter
            loss = loss + self.simple_loss(x, t)

        loss = loss / self.timesteps
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        loss = self.simple_loss(x, t)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.simple_loss_full(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.simple_loss_full(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer with Cosine Annealing Scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99))
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.lr * 0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    @torch.no_grad()
    def sample(self, shape):
        """Generates new Fourier coefficient samples from Gaussian noise."""
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.timesteps)):
            print(t)
            # Convert timestep into batch tensor
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)

            # Compute model prediction
            predicted_noise = self(x, t_tensor)  # Predicted noise at timestep t

            # Compute α_t and noise scale
            alpha_bar = self.alpha_cumprod[t].view(-1, 1, 1).to(self.device)
            alpha = self.alphas[t].view(-1, 1, 1).to(self.device)

            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)  # No noise when t=0

            # Reverse diffusion.py step
            x = (x - ((1 - alpha) * predicted_noise / torch.sqrt(1 - alpha_bar))) / torch.sqrt(alpha)
            x = x + torch.sqrt(1 - alpha) * noise

        return x


class ConditionalDiffusionModel(pl.LightningModule):
    """Diffusion Model"""

    def __init__(self, nnet, timesteps=1000, beta_min=1e-4, beta_max=0.02, lr=1e-4, sequence_length=128, **kwargs):
        super().__init__()
        self.nnet = nnet
        self.lr = lr
        # Define noise schedule
        self.timesteps = timesteps
        betas = torch.linspace(beta_min, beta_max, timesteps, device=self.device)
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.T_max = kwargs.get("T_max", 100)

    def forward(self, x, b, t):
        return self.nnet(x, b, t)

    def q_sample(self, x_start, b, t):
        """Adds Gaussian noise to the input signal according to the diffusion.py schedule."""
        noise = torch.randn_like(x_start)
        self.alpha_cumprod = self.alpha_cumprod.to(x_start.device)
        alpha_bar_t = self.alpha_cumprod[t].view(-1, 1, 1).to(x_start.device)
        # modify only selected dimensions!
        mean_t = torch.sqrt(alpha_bar_t) * x_start * b
        noise_t = noise * b
        x_t = mean_t + torch.sqrt(1 - alpha_bar_t) * noise_t
        return x_t, noise_t

    def simple_loss(self, x, b, t):
        x_noisy, noise = self.q_sample(x, b, t)
        predicted_noise = self.forward(x_noisy, b, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def simple_loss_full(self, batch, batch_idx):
        x, b = batch
        batch_size = x.shape[0]
        loss = 0.

        for t_iter in range(self.timesteps):
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device) * 0 + t_iter
            loss = loss + self.simple_loss(x, b, t)

        loss = loss / self.timesteps
        return loss

    def training_step(self, batch, batch_idx):
        x, b = batch
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        loss = self.simple_loss(x, b, t)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.simple_loss_full(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.simple_loss_full(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer with Cosine Annealing Scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99))
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.lr * 0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    @torch.no_grad()
    def sample(self, shape, b):
        """Generates new Fourier coefficient samples from Gaussian noise."""
        x = torch.randn_like(b, device=self.device)
        x = x * b

        for t in reversed(range(self.timesteps)):
            print(t)
            # Convert timestep into batch tensor
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)

            # Compute model prediction
            predicted_noise = self.forward(x, b, t_tensor)  # Predicted noise at timestep t

            # Compute α_t and noise scale
            alpha_bar = self.alpha_cumprod[t].view(-1, 1, 1).to(self.device)
            alpha = self.alphas[t].view(-1, 1, 1).to(self.device)

            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)  # No noise when t=0
            noise = noise * b

            # Reverse diffusion.py step
            x = (x - ((1 - alpha) * predicted_noise / torch.sqrt(1 - alpha_bar))) / torch.sqrt(alpha)
            x = x + torch.sqrt(1 - alpha) * noise

        return x


class BinaryDiffusion(pl.LightningModule):
    """Diffusion Model with Analog Bits (only 1 bit per dimension)"""

    def __init__(self, nnet, timesteps=1000, beta_min=1e-4, beta_max=0.02, lr=1e-4, sequence_length=128, **kwargs):
        super().__init__()
        self.nnet = nnet
        self.lr = lr
        # Define noise schedule
        self.timesteps = timesteps
        betas = torch.linspace(beta_min, beta_max, timesteps, device=self.device)
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.T_max = kwargs.get("T_max", 100)

    def forward(self, x, t):
        return self.nnet(x, t)

    def q_sample(self, x_start, t):
        """Adds Gaussian noise to the input signal according to the diffusion.py schedule."""
        noise_t = torch.randn_like(x_start)
        self.alpha_cumprod = self.alpha_cumprod.to(x_start.device)
        alpha_bar_t = self.alpha_cumprod[t].view(-1, 1, 1).to(x_start.device)
        # modify only selected dimensions!
        mean_t = torch.sqrt(alpha_bar_t) * x_start
        x_t = mean_t + torch.sqrt(1 - alpha_bar_t) * noise_t
        return x_t, noise_t

    def simple_loss(self, x, t):
        x_noisy, noise = self.q_sample(x, t)
        predicted_noise = self.forward(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def simple_loss_full(self, batch, batch_idx):
        _, x = batch
        x = 2. * x - 1. # {0, 1} -> {-1, 1}
        batch_size = x.shape[0]
        loss = 0.

        for t_iter in range(self.timesteps):
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device) * 0 + t_iter
            loss = loss + self.simple_loss(x, t)

        loss = loss / self.timesteps
        return loss

    def training_step(self, batch, batch_idx):
        _, x = batch
        x = 2. * x - 1.  # {0, 1} -> {-1, 1}
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        loss = self.simple_loss(x, t)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.simple_loss_full(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.simple_loss_full(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer with Cosine Annealing Scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99))
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.lr * 0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    @torch.no_grad()
    def sample(self, shape):
        """Generates new Fourier coefficient samples from Gaussian noise."""
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.timesteps)):
            print(t)
            # Convert timestep into batch tensor
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)

            # Compute model prediction
            predicted_noise = self.forward(x, t_tensor)  # Predicted noise at timestep t

            # Compute α_t and noise scale
            alpha_bar = self.alpha_cumprod[t].view(-1, 1, 1).to(self.device)
            alpha = self.alphas[t].view(-1, 1, 1).to(self.device)

            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)  # No noise when t=0

            # Reverse diffusion.py step
            x = (x - ((1 - alpha) * predicted_noise / torch.sqrt(1 - alpha_bar))) / torch.sqrt(alpha)
            x = x + torch.sqrt(1 - alpha) * noise
            x = torch.clamp(x, -1., 1.)

        return x
