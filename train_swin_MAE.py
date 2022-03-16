# -*- coding: utf-8 -*-
import os
import platform
import random

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch.backends.cudnn
from tensorboardX import SummaryWriter
# import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
# from build_swrepnet import swrepnet
from model_masked_tansformer_basic import MaskedAutoencoder

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# LR = 6e-6
num_epochs = 400
preheat_epoch = 0
BATCH_SIZE = 1
NUM_WORKERS = 0
LOG_DIR = './log/swin_MAE_log0311_0'
CKPT_DIR = './ckpt/ckp_swin_MAE__log0311_0'
new_train = True
lastCkptPath = None
# lastCkptPath = '/public/home/dongsx/rep_proj/swrnet/ckp_repdm_i3d_1/ckpt190_valMAE_1.454.pt'
# lastCkptPath = r'/public/home/zhaoyq/GTRM/cvpr2020/ckp_GTRM_feat_4/ckpt_71_valMAE_1.0.pt'

if platform.system() == 'Windows':
    data_root = r'E:\CS\pose\SWRNET\data\LSP'
    NUM_WORKERS = 0
    lastCkptPath = None
    BATCH_SIZE = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device_ids = [0]
    # device = torch.device('cpu')
    device = torch.device('cuda')
else:
    # data_root = r'/public/home/zhaoyq/LLSP_npz'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

if not os.path.exists(CKPT_DIR):
    os.mkdir(CKPT_DIR)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

from dataset_feature_swin import MyDataset

train_dataset = MyDataset()
train_loader = DataLoader(dataset=train_dataset, pin_memory=False, batch_size=BATCH_SIZE,
                          drop_last=False, shuffle=True, num_workers=NUM_WORKERS)




def adjust_learning_rate(optimizer, curr_epoch,LR,min_lr,epochs,warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if curr_epoch < warmup_epochs:
        lr = LR * curr_epoch / warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (curr_epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    # device = 'cpu'
    model = MaskedAutoencoder()
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = MMDataParallel(model, device_ids=device_ids)
    model.to(device)

    # tensorboard
    new_train = True
    log_dir = LOG_DIR
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)
    if new_train is True:
        del_list = os.listdir(log_dir)
        for f in del_list:
            file_path = os.path.join(log_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    writer = SummaryWriter(log_dir=log_dir)
    # ipdb.set_trace()
    if lastCkptPath is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        currEpoch = checkpoint['epoch']
        trainLosses = checkpoint['trainLoss']
        validLosses = checkpoint['valLoss']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        currTrainstep = checkpoint['train_step']
        currValidationstep = checkpoint['valid_step']
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
    else:
        currEpoch = 0
        currTrainstep = 0
        currValidationstep = 0
    # criterion1 = nn.MSELoss().to(device)
    scaler = GradScaler()
    # optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': 1e-5}], lr=LR)
    param_groups = optim_factory.add_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5, betas=(0.9, 0.95))
    # scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 40, T_mult=2,)
    # print("********* Training begin *********")
    print(currEpoch)
    ep_pbar = tqdm(range(currEpoch, num_epochs))
    train_step = currTrainstep
    valid_step = currValidationstep
    for epoch in ep_pbar:
        # save evaluation metrics
        train_loss = []

        pbar = tqdm(train_loader, total=len(train_loader))
        batch_idx = 0
        model.train()
        data_iter_step = 0
        for datas,video_name in pbar:
            datas = datas.to(device)
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, epochs=num_epochs,LR=1.5e-5,min_lr=0,warmup_epochs=40)
            data_iter_step+=1
            # 向前传播
            with autocast():
                loss, pred, mask = model(datas)

            scaler.scale(loss.float()).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(float(loss))

            pbar.set_postfix({'Epoch': epoch,
                              'loss': float(np.mean(train_loss)),
                              'lr': optimizer.param_groups[0]["lr"]})

        # per epoch of train and valid over
        ep_pbar.set_postfix({'Epoch': epoch,
                             'loss': float(np.mean(train_loss)),
                             })

        # tensorboardX  per epoch

        writer.add_scalars('train/Loss',
                           {"train_Loss": float(np.mean(train_loss))}, epoch)
        writer.add_scalars('train/learning rate', {"learning rate": optimizer.param_groups[0]["lr"]},
                           epoch)

        # save model weights

        saveCkpt = False
        ckpt_name = '/ckpt'
        if saveCkpt  or epoch % 20 == 0 or epoch + 1  == num_epochs  :

            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLoss': float(np.mean(train_loss)),
                'train_step': train_step,

            }
            torch.save(checkpoint,
                       CKPT_DIR + ckpt_name + str(epoch) +
                       str(float(np.mean(train_loss))) + '.pt')
    writer.close()
