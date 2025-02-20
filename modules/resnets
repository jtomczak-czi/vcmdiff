import torch
import torch.nn as nn

from modules.nn_modules import (TimeEmbedding, ConvBlock, TimestepEmbedder, PositionalEncoding, DiTBlock, FinalLayer,
                                ResBlock, MLPblock, timestep_embedding)

####################
# linear ResNet
####################
class ResNet(nn.Module):
    def __init__(self, in_features: int, mid_features: int, num_blocks):
        super().__init__()

        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()

        for _ in range(self.num_blocks):
            self.blocks.append(ResBlock(in_features, mid_features))

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x


class ResNetTime(nn.Module):
    def __init__(self, in_features: int, mid_features: int, out_features: int, num_blocks: int, num_steps: int):
        super().__init__()

        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()

        for i in range(self.num_blocks):
            if i == 0:
                self.blocks.append(ResBlock(in_features, mid_features, out_features, final_activation=nn.Tanh()))
            elif i ==  (self.num_blocks - 1):
                self.blocks.append(ResBlock(out_features, mid_features, in_features, final_activation=nn.Tanh()))
            else:
                self.blocks.append(ResBlock(out_features, mid_features, out_features, final_activation=nn.Identity()))

        # self.embed = nn.Embedding(num_steps, in_features)
        self.embed = timestep_embedding(torch.arange(num_steps), out_features, max_period=10000)

    def forward(self, x, t):
        x = x.squeeze()
        h = self.embed[t.long()]

        for id, block in enumerate(self.blocks):
            x = block(x) if id == 0 else block(x + h)

        return x.unsqueeze(1)