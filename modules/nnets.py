import torch
import torch.nn as nn

from modules.nn_modules import (TimeEmbedding, ConvBlock, TimestepEmbedder, PositionalEncoding, DiTBlock, FinalLayer,
                                ResBlock, MLPblock, timestep_embedding)


class MLPTime(nn.Module):
    def __init__(self, in_features: int, mid_features: int, num_blocks: int, num_steps: int):
        super().__init__()

        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()

        for i in range(self.num_blocks):
            if i == 0:
                self.blocks.append(MLPblock([in_features, mid_features]))
            elif i ==  (self.num_blocks - 1):
                self.blocks.append(MLPblock([mid_features, in_features]))
            else:
                self.blocks.append(MLPblock([mid_features, mid_features]))

        # self.embed = nn.Embedding(num_steps, in_features)
        self.time_embed = timestep_embedding(torch.arange(num_steps), mid_features, max_period=10000)

    def forward(self, x, t):
        # flatten
        C = x.shape[1]
        x = x.flatten(1)
        # time embedding
        h = self.time_embed[t.long()]

        for id, block in enumerate(self.blocks):
            x = block(x) if (id == 0) or (id == (len(self.blocks) - 1)) else block(x + h)

        return x.view(x.shape[0], C, -1)