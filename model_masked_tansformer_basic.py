import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Block


class MaskedAutoencoder(nn.Module):

    def __init__(self, in_chans=1024,
                 embed_dim=1024, encoder_depth=12, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, max_len=1000):
        super().__init__()
        self.max_len = max_len
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # No class token
        self.encoder_embed = nn.Linear(in_chans, embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, embed_dim), requires_grad=False)

        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])  # transformer encoder
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_len, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans, bias=True)  # decoder to feature
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

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

        decoder_pos_embed = self.get_1d_sincos_pos_embed(d_model=self.decoder_pos_embed.shape[2],
                                                         max_len=self.decoder_pos_embed.shape[1])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch_size, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)  # torch.argsort 返回排序后的索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 元素按从小达到排序，

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # 收集输入的特定维度指定位置的数值

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x:[10,t,1024,]

        x = x + self.pos_embed[:, :x.size(1), :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # x.shape [batch_size, length, dim]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed[:, :x.size(1), :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, target, pred_feature, mask):
        """
        input_feature: [batchsize, length, channel] is target
        pred_feature: [batchsize, length, channel]
        mask: [batch_size, length], 0 is keep, 1 is remove,
        """
        loss = (pred_feature - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per frame

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed feature

        return loss

    def forward(self, x, mask_ratio=0.25):
        '''
        feature: [10,1024,t]
        x : [10,1024,t] -> input size [batch_size, length, dim]
        '''
        x = x.transpose(1, 2)  # ->[10,t,1024]
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, f_size*f_size*channel]
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

# t = MaskedAutoencoder()
# x = torch.rand([2,1024,6])
# loss, pred, mask = t(x)
# print(loss)