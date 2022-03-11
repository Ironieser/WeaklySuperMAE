

import math
import torch


def random_masking( x, mask_ratio):
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
    ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # 收集输入的特定维度指定位置的数值
    return x_masked, mask, ids_restore
x = torch.rand([2,5,1024]).to('cpu')
random_masking(x,0.25)
