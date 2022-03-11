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
    def __init__(self, transforms=None,mode = 'test'):

        super().__init__()
        self.mode = mode
        if self.mode == 'train':
            self.root_dir = '/public/home/dongsx/wsvad/data/shanghaitech/training/frames_bgr/'
        elif self.mode == 'test':
            self.root_dir = '/public/home/dongsx/wsvad/data/shanghaitech/testing/frames/'
        self.image_size = 224
        self.width = 320
        self.height = 240
        self.data_norm = False
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
        if self.data_norm :
            means = [0.472, 0.477, 0.477]
            stds = [0.188, 0.178, 0.186]
        else:
            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
        if self.read_img_tool == 'cv2':
            # input ->[t,c,h,w]
            return  T.Compose([
                # T.Resize([240, 320]),
                T.TenCrop(224),  # ->[c,h,w]
                T.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                T.Normalize(mean=means, std=stds)
            ])
        else:
            return  T.Compose([
                T.Resize([240, 320]),
                T.TenCrop(224),  # ->[c,h,w]
                T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),

                T.Normalize(mean=means, std=stds)
            ])

    def __getitem__(self, index):
        # ipdb.set_trace()
        frames = self.read_img(index)
        frames_tensor = None
        if self.transforms is not None:
            for frame in frames:
                if frames_tensor is None:
                    frames_tensor = self.transforms(frame).unsqueeze(0)
                else:
                    frames_tensor = torch.cat((frames_tensor, self.transforms(frame).unsqueeze(0)), 0)

        frames_tensor = frames_tensor.permute(1, 2, 0, 3, 4)
        # frames_tensor = torch.rand([10,3,16,224,224])
        return frames_tensor,self.clip_list[index].split('+')[0]  # ->[ncrops,c, t, h, w]

    def __len__(self):
        """返回数据集的大小"""
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
        config = r'/public/home/dongsx/wsvad/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
        pretrain_ckpt = r'/public/home/dongsx/wsvad/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
        self.swin = self.load_model(config, pretrain_ckpt)
        self.pool3d = torch.nn.AvgPool3d([8, 7, 7])

    def forward(self, x, components=None):
        # batch size should be 1
        # input video [1,ncrops,c, t, h, w]
        # ipdb.set_trace()
        b, ncrops, c, t, h, w = x.size()
        x = x.view(-1, c, t, h, w)
        with torch.no_grad():
            if t % 16 != 0:
                temp = torch.zeros([b*ncrops, c, 16 - t % 16, h,w]).to(x.device)
                x = torch.cat((x, temp), 2)
            x = self.swin(x) # ->[b,1024,t/2,7,7]
            x = self.pool3d(x) # ->[b,1024,t/16,1,1]
        x = x.reshape(b*ncrops, 1024, -1)  # ->[b,1024,t/16]
        return x

    def load_model(self, config, checkpoint):
        """加载预训练模型"""
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location='cpu')
        print('--------- backbone loading finished ------------')
        return model.backbone



if __name__ == '__main__':
    train_dataset = MyDataset()
    train_loader = DataLoader(dataset=train_dataset, pin_memory=True, batch_size=1, drop_last=False, shuffle=False,
                              num_workers=16)
    # device = 'gpu'
    model = swin().to(device)
    # x = torch.rand([1,10,3,84,224,224]).to(device)

    model.eval()
    # # y = model(x)
    # print(y.size())
    print('------------extracting train-----------------')
    with torch.no_grad():
        pbar = tqdm(train_loader, total=len(train_loader))
        file_tensor = None
        file_tag = None
        for data,filename in pbar:
            # feature: [b,ncrops,c, t, h, w]
            data = data.to(device)
            # ipdb.set_trace()
            feature = model(data) # ->feature[b*ncrops,2048,1]
            if file_tensor is None:
                file_tag = filename[0]
                file_tensor = feature
            elif filename[0] != file_tag and file_tensor is not None:
                print(file_tag, 'save finished: size is',file_tensor.size())
                # torch.save(file_tensor, '/public/home/dongsx/wsvad/data/shanghaitech/training/feature_swin_16F_norm/' + file_tag + '.pt')
                torch.save(file_tensor, '/public/home/dongsx/wsvad/data/shanghaitech/testing/feature_swin_16F/' + file_tag + '.pt')
                file_tag = filename[0]
                file_tensor = feature
            else:
                file_tensor = torch.cat((feature,file_tensor),2)
        if file_tensor is not None and file_tag is not None:
            print(file_tag, 'save finished: size is',file_tensor.size())
            # torch.save(file_tensor,'/public/home/dongsx/wsvad/data/shanghaitech/training/feature_swin_16F_norm/' + file_tag + '.pt')
            torch.save(file_tensor,'/public/home/dongsx/wsvad/data/shanghaitech/testing/feature_swin_16F/' + file_tag + '.pt')
    print('------------extracting over-----------------')
