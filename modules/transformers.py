import torch
import torch.nn as nn

from modules.nn_modules import (TimeEmbedding, ConvBlock, TimestepEmbedder, PositionalEncoding, DiTBlock, FinalLayer,
                                ResBlock, MLPblock, timestep_embedding)


####################
# DiT
####################
class DiT(nn.Module):
    """Diffusion Transformer for 1D Fourier Coefficients with Positional Encoding."""
    def __init__(self, sequence_length=128, in_channels=2, hidden_size=512, depth=4, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embedding = PositionalEncoding(sequence_length, hidden_size)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, in_channels)

    def forward(self, x, t):
        x = x.permute(0, 2, 1)  # Convert to (batch, seq, channels)
        x = self.input_proj(x)  # Project to hidden space
        x = self.pos_embedding(x)  # Add positional encoding

        t_emb = self.t_embedder(t)  # Compute time embedding

        for block in self.blocks:
            x = block(x, t_emb)

        return self.final_layer(x, t_emb).permute(0, 2, 1)  # Convert back to (batch, channels, seq)