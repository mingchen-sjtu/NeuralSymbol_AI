#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2022

@author: WangWenshuo
"""
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(root_path)

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import dataloader
from dataloader import BatteryDissembleImageDataset
from importlib.machinery import SourceFileLoader
import algorithms as alg

from dataset_generation import *
from params import *
from utils import file_path, load_vae_model, enocde_dataset

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# llustration of VAE trained with extended loss function on training(left) and testing(right) dataset.
def plot_distance():

	# load VAE model
    # config_file="VAE_UnityStacking_L1_dual_branch"
    # checkpoint_file="vae_lastCheckpoint.pth"
    config_file="VAE_ShirtFolding_L1"
    checkpoint_file="vae_best_checkpoint.pth"
    channels_num = 3

    #load VAE
    vae_config_file = os.path.join('.', 'configs', config_file + '.py')
    vae_directory = os.path.join('.', 'models', checkpoint_file)
    vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config 
    # print('vae_config: ', vae_config)
    vae_config['exp_name'] = config_file
    vae_config['vae_opt']['exp_dir'] = vae_directory  
    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    vae_algorithm.load_checkpoint('models/' + config_file + "/" + checkpoint_file)
    vae_algorithm.model.eval()

    # action_dist = vae_algorithm.epoch_action_dist_list # inter distance of training set
    # noaction_dist = vae_algorithm.epoch_noaction_dist_list # intra distance of training set
    action_dist = vae_algorithm.test_action_dist_list # inter distance of testing set
    noaction_dist = vae_algorithm.test_noaction_dist_list # intra distance of testing set


	# plot distance
    fig, ax = plt.subplots()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.set_facecolor("whitesmoke")

    # for epoch in range(1, len(action_dist)):
    # 	plt.scatter(epoch, action_dist[epoch], s=15, marker='^', color='blue')
    # 	plt.scatter(epoch, noaction_dist[epoch], s=15, marker='*', color='red')

    epochs = range(1, len(action_dist))
    plt.scatter(epochs, action_dist[1:], s=15, marker='^', color='blue', label='inter distance')
    plt.scatter(epochs, noaction_dist[1:], s=15, marker='*', color='red', label='intra distance')

    # plt.axhline(23.5, ls="-.", lw=1, c='orange')
    # plt.axhline(15.5, ls="-.", lw=1, c='orange')
    # plt.axhline(12.5, ls="-.", lw=1, c='orange')
    # plt.axhline(5.0, ls="-.", lw=1, c='orange')
    plt.axhline(23.5, ls="-.", lw=1, c='orange')
    plt.axhline(15.6, ls="-.", lw=1, c='orange')
    plt.axhline(7.5, ls="-.", lw=1, c='orange')

    plt.legend(loc='center right')
    plt.grid(color='white')

    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('distance', fontsize=12)
    plt.show()


# Incorrect sequences versus k
def plot_cluster_num():
    
    file_path = os.path.join(dataset_path, 'action_sequence_test.txt')
    num_sequences, images_paths, GT_state_nums = read_sequence_txt(txt_file_path=file_path)

    min_num, max_num = 1, 11 # actual 1~10
    count_list = [609, 391, 246, 161, 139, 109, 93, 71, 63, 61]
    y = np.asarray(count_list) / np.asarray(images_paths, dtype=object).shape[0]

    # plot & save figure
    fig, ax = plt.subplots()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.set_facecolor("whitesmoke")

    plt.rcParams.update({'font.size': 20})
    plt.plot(np.arange(min_num, max_num), np.around(y, 2), marker='*', markersize='7', color='red')

    plt.axhline(5, ls="-.", lw=1, c='orange')
    plt.axhline(0, ls="-.", lw=1, c='orange')

    plt.xlabel('cluster num', fontsize=12)
    plt.ylabel('Incorrect Rate', fontsize=12)
    # plt.title('loss accuracy')

    plt.grid(color='white')
    plt.show()


# Success rate versus k
def plot_SR():
	First_success_rate = [0.299, 0.5525, 0.7895, 0.7855, 0.3525, 0.3555, 0.327, 0.335]
	Overall_success_rate = [0.504, 0.9525, 0.9505, 0.9465, 0.987, 0.991, 0.99, 0.985]

	# plot & save figure
	fig, ax = plt.subplots()
	ax.spines['left'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.tick_params(bottom=False, top=False, left=False, right=False)
	ax.set_facecolor("whitesmoke")

	plt.rcParams.update({'font.size': 15})
	plt.plot(np.arange(2, 10), First_success_rate, marker='s', markersize='5', color='orangered', label='First success rate')
	plt.plot(np.arange(2, 10), Overall_success_rate, marker='D', markersize='4.5', color='royalblue', label='Overall success rate')

	plt.axhline(1, ls="-.", lw=1, c='orange')

	plt.xlabel('cluster num')
	plt.ylabel('Success rate')
	plt.legend(fontsize='medium', loc='center right', bbox_to_anchor=(1.0, 0.79))

	plt.grid(color='white')
	plt.show()



if __name__ == '__main__':

	# plot_distance()

	# plot_cluster_num()

	plot_SR()
