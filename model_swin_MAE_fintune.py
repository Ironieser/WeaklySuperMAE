import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Block
from model_masked_tansformer_basic import MaskedAutoencoder
class swin_mae_finetune(nn.Module):

    def __init__(self,  embed_dim=1024,max_len=1000):
        super().__init__()
        self.max_len = max_len
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        backbone = MaskedAutoencoder()
        self.encoder = nn.ModuleList(list(backbone.children())[:3])
        # self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder = nn.ModuleList(list(backbone.children())[3:])
        # --------------------------------------------------------------------------


        self.initialize_weights()

    def get_1d_sincos_pos_embed(self, d_model=1024, max_len=1000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # -> [1,max_len,d_model]
        return pe

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = self.get_1d_sincos_pos_embed(d_model=self.pos_embed.shape[2],
                                                 max_len=self.pos_embed.shape[1])
        self.pos_embed.data.copy_(pos_embed)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, mask_ratio=0.25):
        '''
        feature: [1,10,1024,t]
        x : [10,1024,t] -> input size [batch_size, length, dim]
        '''
        x = x.reshape(10,1024,-1)
        x = x.transpose(1, 2)  # ->[10,t,1024]
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, f_size*f_size*channel]
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask
