import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    )#.to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ExpActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class BiasLayer(nn.Module):
    def __init__(self, out_features, data=None):
        super().__init__()
        if data is None:
            self.bias = torch.FloatTensor(torch.randn(out_features))
        else:
            self.bias = torch.FloatTensor(data)
        self.bias.requires_grad = True

    def forward(self, x):
        return x + self.bias


class MLPblock(nn.Module):
    def __init__(self, num_features=[100, 100,], activation=nn.ReLU()):
        super().__init__()
        assert len(num_features) >= 2
        self.num_features = num_features

        self.layers = nn.ModuleList()
        for i in range(len(self.num_features) - 1):
                self.layers.append(nn.Sequential(nn.Linear(num_features[i], num_features[i + 1]), activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


#################################################################################
#                             linear ResNet                                     #
#################################################################################
class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ResBlock(nn.Module):
    '''
    Based on ConvNext v2: https://arxiv.org/pdf/2301.00808
    '''
    def __init__(self, in_features: int, mid_features: int, out_features: int, final_activation: nn.Module):
        super().__init__()

        # hyperparams
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features

        # layers
        self.linear_1 = nn.Linear(in_features, mid_features)
        self.ln = nn.LayerNorm(mid_features)
        self.linear_2 = nn.Linear(mid_features, mid_features)
        self.gelu = nn.GELU()
        self.grn = GRN(mid_features)

        self.final_activation = final_activation

        if out_features != in_features:
            self.proj = nn.Linear(in_features, out_features)
            self.linear_3 = nn.Linear(mid_features, out_features)
        else:
            self.linear_3 = nn.Linear(mid_features, in_features)

    def forward(self, x):
        h = self.linear_1(x)
        h = self.ln(h)
        h = self.linear_2(h)
        h = self.gelu(h)
        h = self.grn(h)
        h = self.linear_3(h)

        if self.in_features == self.out_features:
            return self.final_activation(x + h)
        else:
            return self.final_activation(self.proj(x) + h)

class ResBlockConv1d(nn.Module):
    '''
    Based on ConvNext v2: https://arxiv.org/pdf/2301.00808
    '''
    def __init__(self, in_features: int, mid_features: int, out_features: int):
        super().__init__()

        # hyperparams
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features

        # layers
        self.linear_1 = nn.Conv1d(in_channels=in_features, out_channels=mid_features)
        self.ln = nn.LayerNorm(mid_features)
        self.linear_2 = nn.Conv1d(in_channels=mid_features, out_channels=mid_features)
        self.gelu = nn.GELU()
        self.grn = GRN(mid_features)

        if out_features != in_features:
            self.proj = nn.Linear(in_features, out_features)
            self.linear_3 = nn.Linear(mid_features, out_features)
        else:
            self.linear_3 = nn.Linear(mid_features, in_features)

    def forward(self, x):
        h = self.linear_1(x)
        h = self.ln(h)
        h = self.linear_2(h)
        h = self.gelu(h)
        h = self.grn(h)
        h = self.linear_3(h)

        if self.in_features == self.out_features:
            return x + h
        else:
            return self.self.proj(x) + h


#################################################################################
#                             conv UNet 1d                                      #
#################################################################################

class TimeEmbedding(nn.Module):
    """Sinusoidal Timestep Embedding"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        half_dim = self.embed_dim // 2
        exp = torch.exp(-torch.arange(half_dim, dtype=torch.float32, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        sinusoid_in = t[:, None] * exp
        emb = torch.cat([torch.sin(sinusoid_in), torch.cos(sinusoid_in)], dim=-1)
        return self.linear(emb)


class TimeAwareSelfAttention(nn.Module):
    """Self-Attention Mechanism with Timestep-Aware Query Injection"""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, t_emb):
        attn_output, _ = self.attn(x, x, x)  # Self-attention
        return attn_output


class ConvBlock(nn.Module):
    """1D Convolutional Block with Stricter Downsampling and Timestep-Aware Attention"""
    def __init__(self, in_channels, out_channels, embed_dim, use_attention=False, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1  # Stricter downsampling

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()
        self.use_attention = use_attention

        # Timestep embedding projection
        self.time_proj = nn.Linear(embed_dim, out_channels)

        # Optional Self-Attention
        if self.use_attention:
            self.attn = TimeAwareSelfAttention(out_channels)

    def forward(self, x, t_emb):
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Convert to (batch, seq, channels) for LayerNorm
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # Convert back to (batch, channels, seq)

        # Inject timestep embedding
        t_emb = self.time_proj(t_emb).unsqueeze(-1)
        x = x + t_emb

        if self.use_attention:
            x = x.permute(0, 2, 1)  # Convert to (batch, seq, channels)
            x = self.attn(x, t_emb)
            x = x.permute(0, 2, 1)

        return self.act(x)


#################################################################################
#                                      DiT                                      #
#################################################################################

def modulate(x, shift, scale):
    """Applies adaptive layer normalization modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Creates sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Embedding for Sequence Order Awareness."""
    def __init__(self, sequence_length, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(sequence_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1)]

class MLP(nn.Module):
    """Feedforward Network for Transformer Blocks."""
    def __init__(self, embed_dim, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion)
        self.fc2 = nn.Linear(embed_dim * expansion, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """DiT Transformer Block with AdaLN-Zero and Positional Embedding."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, expansion=mlp_ratio)

        # AdaLN-Zero conditioning (Adaptive Layer Normalization)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        qkv = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(qkv, qkv, qkv)[0]
        qkv = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(qkv)
        return x


class FinalLayer(nn.Module):
    """Final layer for DiT, mapping features back to Fourier space."""
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)