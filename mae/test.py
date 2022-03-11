
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class tests():
    def __init__(self,):
        self.pos_embed = nn.Parameter(torch.zeros(1,128, 1024), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 128,
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    def __call__(self):
        print(self.pos_embed)



t = tests()
t()
