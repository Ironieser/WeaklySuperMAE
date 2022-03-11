import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import ipdb
from torchvision import transforms as T
from torch.utils.data import Dataset

import platform
import random

import torch.nn.functional as F

from tqdm import tqdm
class MyDataset(Dataset):
    def __init__(self, debug = False):

        super().__init__()

        self.root_dir = '/public/home/dongsx/wsvad/data/shanghaitech/training/feature_swin_16F'
        self.video_names = os.listdir(self.root_dir)
        self.video_names.sort()
        self.file_dict = {}

        init_file = tqdm(self.video_names, total=len(self.video_names))
        print('------prepare the data to load in memory-----')
        for video_name in init_file:
            file_path = os.path.join(self.root_dir, video_name)
            if debug:
                features = torch.load(file_path,map_location=torch.device('cpu'))
            else:
                features = torch.load(file_path)
            self.file_dict[video_name] = features
        print('------finished-----')


    def __getitem__(self, index):
        video_name = self.video_names[index]
        feature = self.file_dict[video_name]
        return feature,video_name  # ->[b,1024,t]

    def __len__(self):
        """返回数据集的大小"""
        return len(self.clip_list)
data=MyDataset(debug = True)
x = data[1]
print(x)