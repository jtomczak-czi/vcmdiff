import torch
import torch.nn as nn

from modules.nn_modules import (TimeEmbedding, ConvBlock, TimestepEmbedder, PositionalEncoding, DiTBlock, FinalLayer,
                                ResBlock, MLPblock, timestep_embedding)


####################
# conv UNet1D
####################
class UNet1D(nn.Module):
    """U-Net with Stricter Downsampling and Fully Connected Bottleneck"""
    def __init__(self, in_channels=2, base_channels=128, embed_dim=256, bottleneck_dim=512):
        super().__init__()
        self.time_embed = TimeEmbedding(embed_dim)

        # Encoder with Stricter Downsampling
        self.enc1 = ConvBlock(in_channels, base_channels, embed_dim, downsample=False)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, embed_dim, downsample=False)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, embed_dim, downsample=False)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, embed_dim, downsample=False)

        # Bottleneck with Fully Connected Layer
        self.bottleneck_conv = ConvBlock(base_channels * 8, base_channels * 8, embed_dim, downsample=False)
        self.bottleneck_fc = nn.Sequential(
            nn.Linear(base_channels * 8, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, base_channels * 8),
            nn.SiLU(),
        )

        # Decoder
        self.dec4 = ConvBlock(base_channels * 8, base_channels * 4, embed_dim, downsample=False)
        self.dec3 = ConvBlock(base_channels * 4, base_channels * 2, embed_dim, downsample=False)
        self.dec2 = ConvBlock(base_channels * 2, base_channels, embed_dim, downsample=False)
        self.dec1 = nn.Conv1d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        # Encoding with Downsampling
        e1 = self.enc1(x, t_emb)  # Downsampled
        e2 = self.enc2(e1, t_emb)  # Downsampled
        e3 = self.enc3(e2, t_emb)  # Downsampled
        e4 = self.enc4(e3, t_emb)  # Downsampled

        # Bottleneck
        b = self.bottleneck_conv(e4, t_emb)

        # Decoding with Skip Connections
        d4 = self.dec4(b, t_emb) + e3
        d3 = self.dec3(d4, t_emb) + e2
        d2 = self.dec2(d3, t_emb) + e1
        d1 = self.dec1(d2)  # Final conv layer

        return d1


class UNetMLPln(nn.Module):
    """U-Net with MLPs"""
    def __init__(self, in_dim=2, mid_dim=128, bottleneck_dim=512, num_steps=10):
        super().__init__()
        self.time_embed = TimestepEmbedder(in_dim)

        # time projections
        self.time_proj_enc = nn.Linear(in_dim, mid_dim)
        self.time_proj_bottleneck = nn.Linear(in_dim, bottleneck_dim)

        # Encoder with Stricter Downsampling
        # self.ln_e1 = nn.LayerNorm(in_dim)
        self.enc1 = MLPblock([in_dim, mid_dim], activation=nn.SiLU())
        self.ln_e2 = nn.LayerNorm(mid_dim)
        self.enc2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.ln_e3 = nn.LayerNorm(mid_dim)
        self.enc3 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.ln_e4 = nn.LayerNorm(mid_dim)

        # Bottleneck with Fully Connected Layer
        self.bottleneck = MLPblock([mid_dim, bottleneck_dim], activation=nn.SiLU())

        # Decoder
        self.ln_d3 = nn.LayerNorm(bottleneck_dim)
        self.dec3 = MLPblock([bottleneck_dim, mid_dim], activation=nn.SiLU())
        self.ln_d2 = nn.LayerNorm(mid_dim)
        self.dec2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.ln_d1 = nn.LayerNorm(mid_dim)
        self.dec1 = MLPblock([mid_dim, in_dim], activation=nn.Identity())

    def forward(self, x, t):
        # flatten
        x = x.flatten(1)
        # time embedding
        t_emb = self.time_embed(t) # B x D_in
        t_proj_enc = self.time_proj_enc(t_emb) # B x D_mid
        t_proj_bottleneck = self.time_proj_bottleneck(t_emb) # B x D_bottleneck

        # Encoding with Downsampling
        # x = self.ln_e1(x)
        e1 = self.enc1(x + t_emb)  # Downsampled
        e1 = self.ln_e2(e1)
        e2 = self.enc2(e1 + t_proj_enc)  # Downsampled
        e2 = self.ln_e3(e2)
        e3 = self.enc3(e2 + t_proj_enc)  # Downsampled
        e3 = self.ln_e4(e3)

        # Bottleneck
        bottl = self.bottleneck(e3 + t_proj_enc)

        # Decoding with Skip Connections
        bottl = self.ln_d3(bottl)
        d3 = self.dec3(bottl + t_proj_bottleneck) + e2
        d3 = self.ln_d2(d3)
        d2 = self.dec2(d3 + t_proj_enc) + e1
        d2 = self.ln_d1(d2)
        d1 = self.dec1(d2)  # Final layer

        return d1.view(x.shape[0], -1)


class UNetMLP(nn.Module):
    """U-Net with MLPs"""
    def __init__(self, in_dim=2, mid_dim=128, bottleneck_dim=512, num_steps=10):
        super().__init__()
        self.time_embed = timestep_embedding(torch.arange(num_steps), in_dim, max_period=10000)

        # time projections
        self.time_proj_enc = nn.Linear(in_dim, mid_dim)
        self.time_proj_bottleneck = nn.Linear(in_dim, bottleneck_dim)

        # Encoder with Stricter Downsampling
        self.enc1 = MLPblock([in_dim, mid_dim], activation=nn.SiLU())
        self.enc2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.enc3 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())

        # Bottleneck with Fully Connected Layer
        self.bottleneck = MLPblock([mid_dim, bottleneck_dim], activation=nn.SiLU())

        # Decoder
        self.dec3 = MLPblock([bottleneck_dim, mid_dim], activation=nn.SiLU())
        self.dec2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.dec1 = MLPblock([mid_dim, in_dim], activation=nn.Identity())

    def forward(self, x, t):
        # flatten
        C = x.shape[1]
        x = x.flatten(1)
        # time embedding
        t_emb = self.time_embed(t)  # B x D_in
        t_proj_enc = self.time_proj_enc(t_emb) # B x D_mid
        t_proj_bottleneck = self.time_proj_bottleneck(t_emb) # B x D_bottleneck

        # Encoding with Downsampling
        e1 = self.enc1(x + t_emb)  # Downsampled
        e2 = self.enc2(e1 + t_proj_enc)  # Downsampled
        e3 = self.enc3(e2 + t_proj_enc)  # Downsampled

        # Bottleneck
        b = self.bottleneck(e3 + t_proj_enc)

        # Decoding with Skip Connections
        d3 = self.dec3(b + t_proj_bottleneck) + e2
        d2 = self.dec2(d3 + t_proj_enc) + e1
        d1 = self.dec1(d2)  # Final layer

        return d1.view(x.shape[0], C, -1)


class UNetMLPbn(nn.Module):
    """U-Net with batch-norm"""
    def __init__(self, in_dim=2, mid_dim=128, bottleneck_dim=512, num_steps=10):
        super().__init__()
        self.time_embed = timestep_embedding(torch.arange(num_steps), in_dim, max_period=10000)

        # time projections
        self.time_proj_enc = nn.Linear(in_dim, mid_dim)
        self.time_proj_bottleneck = nn.Linear(in_dim, bottleneck_dim)

        # Encoder with Stricter Downsampling
        self.enc1 = MLPblock([in_dim, mid_dim], activation=nn.SiLU())
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.enc2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.bn2 = nn.BatchNorm1d(mid_dim)
        self.enc3 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.bn3 = nn.BatchNorm1d(mid_dim)

        # Bottleneck with Fully Connected Layer
        self.bottleneck = MLPblock([mid_dim, bottleneck_dim], activation=nn.SiLU())

        # Decoder
        self.dec3 = MLPblock([bottleneck_dim, mid_dim], activation=nn.SiLU())
        self.dec2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.dec1 = MLPblock([mid_dim, in_dim], activation=nn.Identity())

    def forward(self, x, t):
        # flatten
        C = x.shape[1]
        x = x.flatten(1)
        # time embedding
        t_emb = self.time_embed[t.long()]  # B x D_in
        t_proj_enc = self.time_proj_enc(t_emb) # B x D_mid
        t_proj_bottleneck = self.time_proj_bottleneck(t_emb) # B x D_bottleneck

        # Encoding with Downsampling
        e1 = self.enc1(x + t_emb)  # Downsampled
        e1 = self.bn1(e1)
        e2 = self.enc2(e1 + t_proj_enc)  # Downsampled
        e2 = self.bn2(e2)
        e3 = self.enc3(e2 + t_proj_enc)  # Downsampled
        e3 = self.bn3(e3)

        # Bottleneck
        b = self.bottleneck(e3 + t_proj_enc)

        # Decoding with Skip Connections
        d3 = self.dec3(b + t_proj_bottleneck) + e2
        d2 = self.dec2(d3 + t_proj_enc) + e1
        d1 = self.dec1(d2)  # Final layer

        return d1.view(x.shape[0], C, -1)

class UNetMLP_new(nn.Module):
    """U-Net """
    def __init__(self, in_dim=2, mid_dim=128, bottleneck_dim=512, num_steps=10):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, in_dim), nn.Tanh())

        # time projections
        self.time_proj_enc = nn.Linear(in_dim, mid_dim)
        self.time_proj_bottleneck = nn.Linear(in_dim, bottleneck_dim)

        # Encoder with Stricter Downsampling
        self.enc1 = MLPblock([in_dim, mid_dim], activation=nn.SiLU())
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.enc2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.bn2 = nn.BatchNorm1d(mid_dim)
        self.enc3 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.bn3 = nn.BatchNorm1d(mid_dim)

        # Bottleneck with Fully Connected Layer
        self.bottleneck = MLPblock([mid_dim, bottleneck_dim], activation=nn.SiLU())

        # Decoder
        self.dec3 = MLPblock([bottleneck_dim, mid_dim], activation=nn.SiLU())
        self.dec2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.dec1 = MLPblock([mid_dim, in_dim], activation=nn.Identity())

    def forward(self, x, t):
        # time embedding
        t_emb = self.time_embed(t)  # B x D_in

        t_proj_enc = self.time_proj_enc(t_emb) # B x D_mid
        t_proj_bottleneck = self.time_proj_bottleneck(t_emb) # B x D_bottleneck

        # Encoding with Downsampling
        e1 = self.enc1(x + t_emb)  # Downsampled
        e1 = self.bn1(e1)
        e2 = self.enc2(e1 + t_proj_enc)  # Downsampled
        e2 = self.bn2(e2)
        e3 = self.enc3(e2 + t_proj_enc)  # Downsampled
        e3 = self.bn3(e3)

        # Bottleneck
        b = self.bottleneck(e3 + t_proj_enc)

        # Decoding with Skip Connections
        d3 = self.dec3(b + t_proj_bottleneck) + e2
        d2 = self.dec2(d3 + t_proj_enc) + e1
        d1 = self.dec1(d2)  # Final layer

        return d1.view(x.shape[0], -1)


class UNetMLPx(nn.Module):
    """U-Net that takes x (counts) and b = 1[x>0]"""
    def __init__(self, in_dim=2, mid_dim=128, bottleneck_dim=512, num_steps=10):
        super().__init__()
        self.time_embed = timestep_embedding(torch.arange(num_steps), in_dim, max_period=10000)

        # time projections
        self.time_proj_enc = nn.Linear(in_dim, mid_dim)
        self.time_proj_bottleneck = nn.Linear(in_dim, bottleneck_dim)

        # Encoder with Stricter Downsampling
        self.enc1 = MLPblock([in_dim, mid_dim], activation=nn.SiLU())
        self.enc2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.enc3 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())

        # Bottleneck with Fully Connected Layer
        self.bottleneck = MLPblock([mid_dim, bottleneck_dim], activation=nn.SiLU())

        # Decoder
        self.dec3 = MLPblock([bottleneck_dim, mid_dim], activation=nn.SiLU())
        self.dec2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.dec1 = MLPblock([mid_dim, in_dim], activation=nn.Identity())

    def forward(self, x, b, t):
        # flatten
        C = x.shape[1]
        x = x.flatten(1)
        b = b.flatten(1)

        # time embedding
        t_emb = self.time_embed[t.long()]  # B x D_in
        t_proj_enc = self.time_proj_enc(t_emb) # B x D_mid
        t_proj_bottleneck = self.time_proj_bottleneck(t_emb) # B x D_bottleneck

        # Encoding with Downsampling
        x_in = (x + t_emb) * b
        e1 = self.enc1(x_in)  # Downsampled
        e2 = self.enc2(e1 + t_proj_enc)  # Downsampled
        e3 = self.enc3(e2 + t_proj_enc)  # Downsampled

        # Bottleneck
        bottl = self.bottleneck(e3 + t_proj_enc)

        # Decoding with Skip Connections
        d3 = self.dec3(bottl + t_proj_bottleneck) + e2
        d2 = self.dec2(d3 + t_proj_enc) + e1
        d1 = self.dec1(d2)  # Final layer

        out = (d1 * b).view(x.shape[0], C, -1)
        return out


class UNetMLPb(nn.Module):
    """U-Net for binary diffusion"""
    def __init__(self, in_dim=2, mid_dim=128, bottleneck_dim=512, num_steps=10):
        super().__init__()
        self.time_embed = timestep_embedding(torch.arange(num_steps), in_dim, max_period=10000)

        # time projections
        self.time_proj_enc = nn.Linear(in_dim, mid_dim)
        self.time_proj_bottleneck = nn.Linear(in_dim, bottleneck_dim)

        # Encoder with Stricter Downsampling
        self.layer_norm_e1 = nn.LayerNorm(in_dim)
        self.enc1 = MLPblock([in_dim, mid_dim], activation=nn.SiLU())
        self.layer_norm_e2 = nn.LayerNorm(mid_dim)
        self.enc2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.layer_norm_e3 = nn.LayerNorm(mid_dim)
        self.enc3 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())

        # Bottleneck with Fully Connected Layer
        self.bottleneck = MLPblock([mid_dim, bottleneck_dim], activation=nn.SiLU())

        # Decoder
        self.layer_norm_d3 = nn.LayerNorm(bottleneck_dim)
        self.dec3 = MLPblock([bottleneck_dim, mid_dim], activation=nn.SiLU())
        self.layer_norm_d2 = nn.LayerNorm(mid_dim)
        self.dec2 = MLPblock([mid_dim, mid_dim], activation=nn.SiLU())
        self.layer_norm_d1 = nn.LayerNorm(mid_dim)
        self.dec1 = MLPblock([mid_dim, in_dim], activation=nn.Identity())

    def forward(self, x, t):
        # flatten
        C = x.shape[1]
        x = x.flatten(1)

        # time embedding
        t_emb = self.time_embed[t.long()]  # B x D_in
        t_proj_enc = self.time_proj_enc(t_emb) # B x D_mid
        t_proj_bottleneck = self.time_proj_bottleneck(t_emb) # B x D_bottleneck

        # Encoding with Downsampling
        x_in = (x + t_emb)
        x_in = self.layer_norm_e1(x_in)
        e1 = self.enc1(x_in)  # Downsampled
        e1 = self.layer_norm_e2(e1)
        e2 = self.enc2(e1 + t_proj_enc)  # Downsampled
        e2 = self.layer_norm_e2(e2)
        e3 = self.enc3(e2 + t_proj_enc)  # Downsampled
        e3 = self.layer_norm_e3(e3)

        # Bottleneck
        bottl = self.bottleneck(e3 + t_proj_enc)

        # Decoding with Skip Connections
        bottl = self.layer_norm_d3(bottl)
        d3 = self.dec3(bottl + t_proj_bottleneck) + e2
        d3 = self.layer_norm_d2(d3)
        d2 = self.dec2(d3 + t_proj_enc) + e1
        d2 = self.layer_norm_d1(d2)
        d1 = self.dec1(d2)  # Final layer

        out = d1.view(x.shape[0], C, -1)
        return out