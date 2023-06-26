#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2021

@author: Duyidong WangWenshuo
"""

from __future__ import print_function
import algorithms as alg
import argparse
import pickle
import torch
import os
from dataloader import *
from dataset_generation import *
from importlib.machinery import SourceFileLoader


parser = argparse.ArgumentParser()
parser.add_argument('--exp_vae', type=str, required=True, default='', 
                    help='config file with parameters of the vae model')
parser.add_argument('--chpnt_path', type=str, default='', 
                    help='path to the checkpoint')
parser.add_argument('--num_workers', type=int, default=0,      
                    help='number of data loading workers')
parser.add_argument('--cuda' , type=bool, default=False, help='enables cuda')

args_opt = parser.parse_args()

# Load VAE config file
vae_config_file = os.path.join('.', 'configs', args_opt.exp_vae + '.py')
vae_directory = os.path.join('.', 'models', args_opt.exp_vae)
if (not os.path.isdir(vae_directory)):
    os.makedirs(vae_directory)

print(' *- Training:')
print('    - VAE: {0}'.format(args_opt.exp_vae))

vae_config = SourceFileLoader(args_opt.exp_vae, vae_config_file).load_module().config 
vae_config['exp_name'] = args_opt.exp_vae
vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
print(' *- Loading experiment %s from file: %s' % (args_opt.exp_vae, vae_config_file))
print(' *- Generated logs, snapshots, and model files will be stored on %s' % (vae_directory))

# Initialise VAE model
vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
print(' *- Loaded {0}'.format(vae_config['algorithm_type']))

root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(root_path, 'crop')

# build xxx_file_list.txt files
# build_battery_dissemble_image_dataset(dataset_path=dataset_path)  # file_list.txt train_file_list.txt test_file_list.txt 
# build_battery_dissemble_image_pair_dataset(dataset_path=dataset_path)

# build dataset from repell and appeal sets
# build_dataset_from_repell_appeal_sets(dataset_path, dataset_size=6000)

# build dataset.train & dataset.test txt files
train_dataset = BatteryDissembleDataset(dataset_path=dataset_path, split='train', channels_num=vae_config['vae_opt']['input_channels'])
test_dataset = BatteryDissembleDataset(dataset_path=dataset_path, split='test', channels_num=vae_config['vae_opt']['input_channels'])

if args_opt.num_workers is not None:
    num_workers = args_opt.num_workers
else:
    num_workers = vae_config_file['vae_opt']['num_workers']

# start training
vae_algorithm.train(train_dataset, test_dataset, num_workers, args_opt.chpnt_path)
