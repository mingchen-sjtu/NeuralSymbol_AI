#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2022

@author: WangWenshuo
"""
import sys
import os
import random
import numpy as np
from scipy.stats import norm
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

from cluster import *
from plan import *
from dataset_generation import *
from params import *
from utils import file_path, load_vae_model, enocde_dataset
from dataloader import *

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/')
dataset_path = os.path.join(root_path, 'true_mul_bolt_crops')


# load VAE model
def load_model():
    
    config_file="VAE_ShirtFolding_L1"
    # checkpoint_file="vae_lastCheckpoint.pth"
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

    return vae_algorithm.model


# encode an image by trained VAE model
def encode_an_image(model, img_path, latent_mean):

    if latent_mean is not None:
        latent_mean = latent_mean
        latent_logvar = torch.zeros(latent_mean.shape, device=device)

    if img_path is not None:
        # image -> Tensor
        transform = transforms.Compose(
                            [transforms.Resize((256, 256)),
                             transforms.ToTensor(),
                             ])
        img = transform(Image.open(img_path))
        img = torch.unsqueeze(img, 0)

        # to device encode
        img = img.to(device)
        latent_mean, latent_logvar = model.encoder(img)
    
    z = model.sample(latent_mean, latent_logvar)
    # latent_vector = z_img.cpu().detach().numpy()

    dec_mean, dec_logvar = model.decoder(z)
    # print('dec_mean: ', dec_mean)
    # print('dec_mean shape: ', dec_mean.shape)

    return dec_mean

# return the mean of latent vectors
def enocde_dataset(model, dataset_type='train', class_nums=4):

    battery_set = BatteryDissembleImageDataset(dataset_path=dataset_path, dataset_type=dataset_type, class_nums=class_nums)
    # battery_set = BatteryDissembleDataset(dataset_path=dataset_path, split='train', channels_num=3)
    dataloader = torch.utils.data.DataLoader(battery_set, batch_size=batch_size, shuffle=True,
                                                        num_workers=0, drop_last=True)

    # image_labels & latent_vectors
    image_labels = np.zeros((0))
    latent_vectors = np.zeros((0, 64))
    for batch_idx, (img, image_label) in enumerate(dataloader):
        img = img.to(device)
        image_labels = np.append(image_labels, image_label.cpu().detach().numpy())

        z_img = model(img, sample_latent=True, latent_code=True)
        latent_vectors = np.append(latent_vectors, z_img.cpu().detach().numpy(), axis=0)

    # latent vectors mean
    latent_vectors_sum = np.zeros((4, 64))
    latent_vectors_num = np.zeros((4))

    for i in range(len(image_labels)):
        latent_vectors_sum[int(image_labels[i]), :] += latent_vectors[i, :]
        latent_vectors_num[int(image_labels[i])] += 1

    latent_vectors_mean = latent_vectors_sum / latent_vectors_num.reshape(4, 1) # shape (4, 64)
    print(latent_vectors_mean[0, :])
    print(latent_vectors_mean.shape)


    return latent_vectors_mean


def plot_grid(image):
    """
    Plots an nxn grid of image of size digit_size. Used to monitor the 
    reconstruction of decoded image.
    """
    input_dim = 256 * 256 * 3
    input_channels = 3
    save_path = os.path.join(root_path, 'reconstruct.jpg')

    digit_size = int(np.sqrt(input_dim / input_channels)) # 256
    figure = np.zeros((digit_size, digit_size, input_channels))

    # decode plot
    digit = image[0].permute(1,2,0).detach().cpu().numpy()
    figure[ 0: digit_size,
            0: digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='bone')
    plt.savefig(save_path)
    plt.clf()
    plt.close()


# reconstruct image given probability
def reconstruct(prob):

    VAE_model = load_model()
    latent_vectors_mean = enocde_dataset(VAE_model)

    enc_mean = np.sum(latent_vectors_mean * prob, axis=0).reshape(1, 64)
    enc_mean = torch.from_numpy(enc_mean).float().cuda()
    dec_mean = encode_an_image(model=VAE_model, img_path=None, latent_mean=enc_mean)

    plot_grid(dec_mean)


def action(path):

    vae_model = load_model()
    # build_npy(model=vae_model, encode_dataset=True, cluster=True)
    # goal_proba = find_end_state_distribution()
    goal_proba = np.load(file_path(file_name=end_state_distribution_npy, file_path=True, split=None))
    # print('goal_proba: ', goal_proba)

    probas_norm = img_2_cluster_proba(vae_model=vae_model, test_image=path)
    # print('probas_norm: ', probas_norm)

    predicted_action, plan, predicted_probs, action_prob_dict = search_plan(init_proba=probas_norm, goal_proba=goal_proba)
    print('plan: ', plan)
    print('predicted_probs: ', predicted_probs)
    print('action_prob_dict: ', action_prob_dict)

    return plan


if __name__ == '__main__':

    # sim
    # action_prob_dict:  {'push': array([0.383, 0.369, 0.037, 0.21 ]), 'mate': array([0.667, 0.063, 0.06 , 0.209])}
    img_path = os.path.join(dataset_path, 'hexagon_bolt',  '00000338_3.png')
    # action(img_path)
    VAE_model = load_model()
    latent_vectors_mean = enocde_dataset(VAE_model)

    bolt='out_hex_bolt'
    fileDir=os.path.join(root_path, 'true_mul_bolt_crops_test', bolt)
    pathDir = os.listdir(fileDir)    #取图片的原始路径
    filenumber=len(pathDir)
    rate=1    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    n=0
    fail=[]
    fail_dis=[]                                
    for i in sample:
        img = Image.open(fileDir+  '/'  + i)
        transform = transforms.Compose(
                            [transforms.Resize((256, 256)),
                             transforms.ToTensor(),
                             ])
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        latent_mean, latent_logvar = VAE_model.encoder(img)
        latent_mean = latent_mean.cpu().detach().numpy()[0]
        print("latent_mean\n",latent_mean)
        dist_bolt= np.zeros((4))
        for j in range(4):
            dist_bolt[j] = np.sqrt(np.sum(np.square(latent_vectors_mean[j] - latent_mean)))
        print("dist_bolt\n",dist_bolt)
        bolt_type={0:"in_hex_bolt",1:"star_bolt",2:"out_hex_bolt",3:"cross_hex_bolt"}
        print("bolt_type:",bolt_type[np.argmin(dist_bolt)])
        if bolt_type[np.argmin(dist_bolt)]==bolt:
            n+=1
        else:
            fail.append(fileDir+  '/'  + i)
            fail_dis.append(dist_bolt)
    print ("success_rate:",float(n/filenumber),"\nnum:",filenumber)
    print ("fail_bolt:",fail,"\nfail_dis:",fail_dis)
    
    



    # transform = transforms.Compose(
    #                         [transforms.Resize((256, 256)),
    #                          transforms.ToTensor(),
    #                          ])
    # img = transform(Image.open(img_path))
    # img = torch.unsqueeze(img, 0)

    # # to device encode
    # img = img.to(device)
    # latent_mean, latent_logvar = VAE_model.encoder(img)
    # latent_mean = latent_mean.cpu().detach().numpy()[0]
    # print("latent_mean\n",latent_mean)
    # dist_bolt= np.zeros((4))
    # for i in range(4):
    #     dist_bolt[i] = np.sqrt(np.sum(np.square(latent_vectors_mean[i] - latent_mean)))
    # print("dist_bolt\n",dist_bolt)
    # bolt_type={0:"m_bolt",1:"cross_bolt",2:"star_bolt",3:"hexagon_bolt"}
    # print("bolt_type:",bolt_type[np.argmin(dist_bolt)])

    # dec_mean = encode_an_image(model=VAE_model, img_path=img_path, latent_mean=None) 
    # print(dec_mean.size())
    # plot_grid(dec_mean)


    # prob = np.asarray([[0, 0, 0, 1.5]]).reshape(4, 1)
    # reconstruct(prob)
    # prob = np.asarray([[1, 0, 0, 0]]).reshape(4, 1)
    # reconstruct(prob)
    