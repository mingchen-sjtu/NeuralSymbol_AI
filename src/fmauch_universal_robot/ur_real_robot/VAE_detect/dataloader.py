#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2021

@author: Duyidong WangWenshuo
"""

from __future__ import print_function
import torch
import torch.utils.data as data
import random
import sys
import pickle
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class ToArray(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return np.asarray(img)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.Tensor(img).unsqueeze(0)


class DepthImageNormilize(object):
    def __init__(self, max_depth_value, min_depth_value):
        self.max_depth_value = float(max_depth_value)
        self.min_depth_value = float(min_depth_value)

    def __call__(self, img):
        return (img - self.min_depth_value) / (self.max_depth_value - self.min_depth_value)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.zeros(tensor.size()).normal_(self.mean, self.std)
        # print('noise: ', noise)
        # print('tensor.add(noise): ', tensor.add(noise))
        return tensor.add(noise)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# ----------------------- #
# --- Build Battery Dissemble Datasets Used in cluster.py --- #
# ----------------------- #
class BatteryDissembleImageDataset(data.Dataset):
    def __init__(self, dataset_path, transform=None, dataset_type='train', channels_num=3, class_nums=4):

        # dataset_txt_path = os.path.join(dataset_path, 'file_list.txt')

        if class_nums == 4:
            dataset_txt_path = os.path.join(dataset_path, 'file_list_{}.txt'.format(dataset_type))
        if class_nums == 11:
            dataset_txt_path = os.path.join(dataset_path, 'file_list_state_{}.txt'.format(dataset_type))

        dataset_txt = open(dataset_txt_path, 'r')
        self._data = dataset_txt.read().split('\n')
        self._data = list(filter(lambda x: not x == '', self._data))
        self._dataset_path = dataset_path
        self.channels_num = channels_num
        self.dataset_name = 'hardcode'

        self._transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        self._depth_transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             ToArray(),
             ToTensor(),
             # DepthImageNormilize(self.max_depth_value, self.min_depth_value)
             ])
    
    def __getitem__(self, index):
        img_path, label = self._data[index].split()
        img_path = os.path.join(self._dataset_path, img_path)
        img = self._transform(Image.open(img_path))
        label = int(label)

        # add depth/mask channel
        if self.channels_num == 4:
            depth_img_path = img_path.replace('rgb', 'mask')
            depth_img = self._depth_transform(Image.open(depth_img_path))
            img = torch.cat((img, depth_img), dim=0)

        return img, label

    def __len__(self):
        return len(self._data)


# ----------------------- #
# --- Custom battery dissemble image pair Datasets --- #
# ----------------------- #
class BatteryDissembleDataset(data.Dataset):
    def __init__(self, dataset_path, split: str='train', transform=None, channels_num=3):
        dataset_txt_path = os.path.join(dataset_path, 'dataset_{}.txt'.format(split))
        dataset_txt = open(dataset_txt_path, 'r')
        self._data = dataset_txt.read().split('\n')
        self._data = list(filter(lambda x: not x == '', self._data))
        self._dataset_path = dataset_path

        self.dataset_name = 'hardcode'
        self.channels_num = channels_num
        self.max_depth_value = 521
        self.min_depth_value = 0
        
        self._transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.ColorJitter(brightness=0.04, contrast=0.08, saturation=0, hue=0),
             # transforms.RandomRotation(degrees=180),
             # transforms.RandomHorizontalFlip(),
             # AddGaussianNoise(0.02, 0.01),

             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # (-1, 1)
             ])

        self._depth_transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             ToArray(),
             # transforms.ToTensor(),
             ToTensor(),
             # DepthImageNormilize(self.max_depth_value, self.min_depth_value)
             ])

    
    def __getitem__(self, index):
        img_1_path, img_2_path, action = self._data[index].split()
        img_1_path = os.path.join(self._dataset_path, img_1_path)
        img_2_path = os.path.join(self._dataset_path, img_2_path)

        img_1 = self._transform(Image.open(img_1_path))
        img_2 = self._transform(Image.open(img_2_path))
        action = int(action)

        # add depth/mask info
        if self.channels_num == 4:
            depth_img_1_path = img_1_path.replace('rgb', 'mask')        
            depth_img_1 = self._depth_transform(Image.open(depth_img_1_path))

            depth_img_2_path = img_2_path.replace('rgb', 'mask')
            depth_img_2 = self._depth_transform(Image.open(depth_img_2_path))
            # print(np.unique(depth_img_2.numpy()))

            img_1 = torch.cat((img_1, depth_img_1), dim=0)
            img_2 = torch.cat((img_2, depth_img_2), dim=0)

        return img_1, img_2, action

    def __len__(self):
        return len(self._data)

    def measure_repeatness(self):
        img_freq = {}
        for i in range(len(self._data)):
            img_1_path, img_2_path, action = self._data[i].split()
            for img_path in [img_1_path, img_2_path]:
                if img_path in list(img_freq.keys()):
                    img_freq[img_path] += 1
                else:
                    img_freq[img_path] = 1
        all_freq = np.array(list(img_freq.values()))
        print('repeatness: ', np.sum(all_freq > 1))


# ----------------------- #
# --- Custom Datasets --- #
# ----------------------- #
class TripletTensorDataset(data.Dataset):
    def __init__(self, dataset_name, split):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
       
        if split == 'test':
            with open('datasets/test_'+self.dataset_name+'.pkl', 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open('datasets/train_'+self.dataset_name+'.pkl', 'rb') as f:
                self.data = pickle.load(f)


    def __getitem__(self, index):
        img1, img2, action = self.data[index]
        return img1, img2, action

    def __len__(self):
        return len(self.data)


class APNDataset(data.Dataset):
    def __init__(self, task_name, dataset_name, split, random_seed, dtype,
                 img_size):
        self.task_name = task_name
        self.dataset_name =  dataset_name
        self.name = dataset_name + '_' + split
        self.split = split.lower()
        self.random_seed = random_seed
        self.dtype = dtype
        self.img_size = img_size

        # Stacking data
        if self.task_name == 'unity_stacking':
            path = 'action_data/{0}/{1}_{2}_seed{3}.pkl'.format(
                    self.dataset_name, self.dtype, self.split, self.random_seed)

            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
                self.data = pickle_data['data']
                self.min, self.max = pickle_data['min'], pickle_data['max']

        # Shirt data
        if self.task_name == 'shirt_folding':
            path = './action_data/{0}/{1}_normalised_{2}_seed{3}.pkl'.format(
                    self.dataset_name, self.dtype, self.split, self.random_seed)

            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
                self.data = pickle_data['data']
                self.min, self.max = pickle_data['min'], pickle_data['max']

    def __getitem__(self, index):
        img1, img2, coords = self.data[index]
        return img1, img2, coords

    def __len__(self):
        return len(self.data)


