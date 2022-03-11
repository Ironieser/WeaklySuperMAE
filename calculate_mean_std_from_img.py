# coding=utf-8
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch.nn.parameter import Parameter
import ipdb
from torchvision import transforms as T
from torch.utils.data import Dataset
import cv2
from PIL import Image
import platform
import random

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, transforms=None):

        super().__init__()

        self.root_dir = '/public/home/dongsx/wsvad/data/shanghaitech/training/frames_bgr/'
        self.image_size = 224
        self.width = 320
        self.height = 240
        self.video_names = os.listdir(self.root_dir)
        self.video_names.sort()
        self.read_img_tool = 'cv2'
        self.transforms = self.trans_func()
        self.file_dict = {}
        self.clip_list = []
        init_file = tqdm(self.video_names, total=len(self.video_names))
        for video_name in init_file:
            file_path = os.path.join(self.root_dir, video_name)
            imgs = os.listdir(file_path)
            imgs.sort()
            for i in range(0,len(imgs),16):
                video_clip = video_name +'+'+ str(i).rjust(3, '0')
                if i+16 < len(imgs):
                    self.file_dict[video_clip] = imgs[i:i+16]
                else:
                    self.file_dict[video_clip] = imgs[i:]
                self.clip_list.append(video_clip)
    def trans_func(self,):
        if self.read_img_tool == 'cv2':
            # input ->[t,c,h,w]
            return  T.Compose([
                # T.Resize([240, 320]),
                T.TenCrop(224),  # ->[c,h,w]
                T.Lambda(lambda crops: torch.stack([crop for crop in crops])),

                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return  T.Compose([
                T.Resize([240, 320]),
                T.TenCrop(224),  # ->[c,h,w]
                T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),

                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        # ipdb.set_trace()
        frames = self.read_img(index) # -> tensor:[t,c,h,w]


        return frames

    def __len__(self):
        return len(self.clip_list)

    def read_img(self, index):
        clip = self.clip_list[index]
        clip_imgs = self.file_dict[clip]

        video_name = clip.split('+')[0]
        video_path = os.path.join(self.root_dir, video_name)
        # imgs = self.file_dict[video_name]
        frames = []
        if self.read_img_tool == 'cv2':
            for img in clip_imgs:
                frame_bgr = cv2.imread(os.path.join(video_path, img))
                frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (self.width,self.height))   #
                frames.append(frame_rgb)
            frames = torch.FloatTensor(frames)  # ->[t,h,w,c]
            frames = frames.permute([0,3,1,2]) # ->[t,c,h,w]
            frames /=255  # -> [t,c,h,w]
            return frames
        else:
            for img in clip_imgs:
                frame_PIL = Image.open(os.path.join(video_path, img))
                frames.append(frame_PIL)
            return frames


class swin(nn.Module):
    def __init__(self, ):
        super(swin, self).__init__()

    def forward(self, x, components=None):
        # batch size should be 1
        # input video -> tensor:[b,t,c,h,w]
        # ipdb.set_trace()
        b, c, t, h, w = x.size()
        x =x.transpose(1,2)
        with torch.no_grad():
            r_mean = x[:, 0, :, :,:].mean()
            g_mean = x[:, 1, :, :,:].mean()
            b_mean = x[:, 2, :, :,:].mean()
            r_std = x[:,0,:,:,:].std()
            g_std = x[:,1,:,:,:].std()
            b_std = x[:,2,:,:,:].std()
        return r_mean,g_mean,b_mean,r_std,g_std,b_std

    def load_model(self, config, checkpoint):
        """加载预训练模型"""
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location='cpu')
        print('--------- backbone loading finished ------------')
        return model.backbone



if __name__ == '__main__':
    train_dataset = MyDataset()
    # print(train_dataset[0])
    train_loader = DataLoader(dataset=train_dataset, pin_memory=True, batch_size=1, drop_last=False, shuffle=False,
                              num_workers=32)
    # device = 'gpu'
    model = swin().to(device)
    # x = torch.rand([1,10,3,84,224,224]).to(device)

    model.eval()
    # # y = model(x)
    # print(y.size())
    r_m, g_m, b_m = [], [], []
    r_s, g_s, b_s = [], [], []
    print('------------extracting train-----------------')
    with torch.no_grad():
        i = 0
        pbar = tqdm(train_loader, total=len(train_loader))
        file_tensor = None
        file_tag = None
        for data in pbar:
            i+=1
            # feature: [b,ncrops,c, t, h, w]
            data = data.to(device)
            # ipdb.set_trace()
            r_mean,g_mean,b_mean,r_std,g_std,b_std = model(data) # ->feature[b*ncrops,2048,1]
            r_m.append(float(r_mean))
            g_m.append(float(g_mean))
            b_m.append(float(b_mean))
            r_s.append(float(r_std))
            g_s.append(float(g_std))
            b_s.append(float(b_std))
            if i%100 == 0:
                rm = np.array(r_m).mean()
                gm = np.array(g_m).mean()
                bm = np.array(g_m).mean()
                rs = np.array(r_s).mean()
                gs = np.array(g_s).mean()
                bs = np.array(b_s).mean()
                print('rm:', rm)
                print('gm:', gm)
                print('bm:', bm)
                print('rs:', rs)
                print('gs:', gs)
                print('bs:', bs)
    print('------------over-----------------')
    print('------------over-----------------')
    rm = np.array(r_m).mean()
    gm = np.array(g_m).mean()
    bm = np.array(g_m).mean()
    rs = np.array(r_s).mean()
    gs = np.array(g_s).mean()
    bs = np.array(b_s).mean()
    print('rm:',rm)
    print('gm:',gm)
    print('bm:',bm)
    print('rs:',rs)
    print('gs:',gs)
    print('bs:',bs)
    print('------------over-----------------')
    print('------------over-----------------')
