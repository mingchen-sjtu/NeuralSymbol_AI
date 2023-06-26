#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2021

@author: DuYidong WangWenshuo
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from progress.bar import Bar
root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(root_path)

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

import dataloader


from dataset_generation import *
from params import *
from utils import file_path, load_vae_model, enocde_dataset, latent_vec_2_cluster_proba


def kmeans_cluster(num, latent_vectors, image_labels):
    # print('*' * 50)
    # print('KMeans Clustering Methods: \n')
    cluster_num = num
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(latent_vectors)

    label_nums = []
    cluster_labels = kmeans.labels_
    cluster_labels_indices = [np.where(cluster_labels == i)[0] for i in np.arange(cluster_num)]

    for index, indices in enumerate(cluster_labels_indices):
        image_label_cluster = image_labels[indices]
        image_label_percentage_cluster = [np.sum(image_label_cluster == i) / image_label_cluster.shape[0] for i in np.arange(label_num)]
        label_nums.append(np.array(image_label_percentage_cluster) * (image_label_cluster.shape[0]))

    np.save(file_path(file_name=label_nums_npy, file_path=False, split=None), np.array(label_nums))
    return kmeans


def cluster_num():

    Transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    GT_state_dict = {'0': 'no_clear_but_no_aim', '1': 'no_clear_but_aim', '2': 'clear_but_no_aim', '3': 'clear_but_aim'}

    file_path = os.path.join(dataset_path, 'action_sequence_test.txt')
    num_sequences, images_paths, GT_state_nums = read_sequence_txt(txt_file_path=file_path)

    # load VAE model
    vae_model = load_vae_model()

    # train dataloader
    latent_vectors_train, image_labels_train = enocde_dataset(vae_model, 'train')

    # for loop cluster nums
    min_num, max_num = 1, 11 # actual 1~10
    count_list = []
    bar = Bar('Processing', max=max_num - min_num)
    for num in range(min_num, max_num):
        cluster_model = kmeans_cluster(num, latent_vectors_train, image_labels_train)

        count = 0
        for i, images_path in enumerate(images_paths):
            images_path_list = []
            for j in range(len(images_path)):
                images_path_list.append(os.path.join(GT_state_dict[str(GT_state_nums[i][j])], 'rgb', images_path[j]))
            if len(images_path_list) == 1:
                continue

            labels = []
            for image_path in images_path_list:
                latent_vectors = np.zeros((0, 64))
                # image tensor -> latent vector
                image_path = os.path.join(dataset_path, image_path)
                img_tensor = Transform(Image.open(image_path))
                img_tensor = img_tensor.expand(1, -1, -1, -1) # (3, 256, 256) -> (1, 3, 256, 256)
                img_tensor = img_tensor.to(device)

                # latent vectors
                latent_vector = vae_model(img_tensor, sample_latent=True, latent_code=True).cpu().detach().numpy().reshape(1, -1) # (1, 3, 256, 256)
                latent_vectors = np.append(latent_vectors, latent_vector, axis=0)

                # cluster model predict
                label = cluster_model.predict(latent_vectors)
                labels.extend(label)

            if len(list(set(labels))) != len(labels):
                count += 1

        # print('count: ', count)
        count_list.append(count)

        bar.next()
    bar.finish()

    print('count_list: ', count_list)

    # plot & save figure
    y = np.asarray(count_list) / np.asarray(images_paths, dtype=object).shape[0]
    plt.plot(np.arange(min_num, max_num), np.around(y, 2), marker='*', markersize='10', color='red')
    plt.xlabel('cluster num')
    plt.ylabel('mistaken ratio')
    plt.title('loss accuracy')
    plt.savefig('./cluster_num_loss.png')


# learning mean and covariance of each cluster
def learn_mean_cov(kmeans_model, train_latent_vectors, train_image_labels):

    cluster_labels = kmeans_model.labels_
    cluster_unique_labels = np.unique(kmeans_model.labels_) # [0 1 2 3]
    cluster_labels_indices = [np.where(cluster_labels == cluster_label)[0] for cluster_label in cluster_unique_labels] # 4 * arrays

    # build connection between image labels and cluster labels: not used
    label_connect_dict = [0] * len(cluster_labels_indices)
    for i in range(len(cluster_labels_indices)):
        train_labels = train_image_labels[cluster_labels_indices[i]].astype(np.int64)
        most_label = np.bincount(train_labels).argmax()
        label_connect_dict[i] = most_label
    # print('label_connect_dict: ', label_connect_dict)

    # obtain cluster mean & covariance
    numpy_means = []
    numpy_covs = []
    for cluster_label in cluster_labels_indices:
        cur_latent_vectors = train_latent_vectors[cluster_label, :]
        # print('cur_latent_vectors: ', cur_latent_vectors.shape)
        numpy_means.append(np.mean(cur_latent_vectors, axis=0))
        numpy_covs.append(np.cov(cur_latent_vectors, rowvar=0))

    numpy_means = np.array(numpy_means)
    numpy_covs = np.array(numpy_covs)

    np.save(file_path(file_name=cluster_means_npy, file_path=False, split=None), numpy_means)
    np.save(file_path(file_name=cluster_covs_npy, file_path=False, split=None), numpy_covs)

    return numpy_means, numpy_covs





# build probability to each cluster
def img_2_cluster_proba(vae_model, test_image, numpy_means=None, numpy_covs=None):

    if isinstance(test_image, str):
        img = Image.open(test_image)
    elif isinstance(test_image, np.ndarray):
        img = Image.fromarray(test_image)

    Transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # image tensor -> latent vector
    img_tensor = Transform(img)
    img_tensor = img_tensor.expand(1, -1, -1, -1) # (3, 256, 256) -> (1, 3, 256, 256)
    img_tensor = img_tensor.to(device)

    # latent vectors
    test_latent_vector = vae_model(img_tensor, sample_latent=True, latent_code=True).cpu().detach().numpy().reshape(1, -1) # (1, 3, 256, 256)

    probas_norm = latent_vec_2_cluster_proba(latent_vector=test_latent_vector, numpy_means=numpy_means, numpy_covs=numpy_covs)

    return probas_norm


def build_npy(model=None, encode_dataset: bool=False, cluster: bool=False, class_nums: int=4):

    if model is None:
        vae_model = load_vae_model()
    else:
        vae_model = model

    if encode_dataset:
        # train enocde_dataset
        latent_vectors_train, image_labels_train = enocde_dataset(vae_model, 'train', class_nums)
        # test dataloader
        latent_vectors_test, image_labels_test = enocde_dataset(vae_model, 'test', class_nums)

        if not os.path.isdir(npy_path):
            os.makedirs(npy_path)

        if class_nums == 4:
            np.save(file_path(file_name=encoded_latent_vectors_npy, file_path=False, split='train'), latent_vectors_train)
            np.save(file_path(file_name=image_labels_npy, file_path=False, split='train'), image_labels_train)

            np.save(file_path(file_name=encoded_latent_vectors_npy, file_path=False, split='test'), latent_vectors_test)
            np.save(file_path(file_name=image_labels_npy, file_path=False, split='test'), image_labels_test)
    else:
        latent_vectors_train = np.load(file_path(file_name=encoded_latent_vectors_npy, file_path=True, split='train'))
        image_labels_train = np.load(file_path(file_name=image_labels_npy, file_path=True, split='train'))

        latent_vectors_test = np.load(file_path(file_name=encoded_latent_vectors_npy, file_path=True, split='test'))
        image_labels_test = np.load(file_path(file_name=image_labels_npy, file_path=True, split='test'))

    if cluster:
        kmeans_train = kmeans_cluster(cluster_center_num, latent_vectors_train, image_labels_train)
        numpy_means, numpy_covs = learn_mean_cov(kmeans_train, latent_vectors_train, image_labels_train)


def test_cluster_accuracy(model=None):

    if model is None:
        vae_model = load_vae_model()
    else:
        vae_model = model

    latent_vectors_test = np.load(file_path(file_name=encoded_latent_vectors_npy, file_path=True, split='test'))
    image_labels_test = np.load(file_path(file_name=image_labels_npy, file_path=True, split='test'))

    probas = []
    for i in range(latent_vectors_test.shape[0]):
        proba = latent_vec_2_cluster_proba(latent_vectors_test[i, :])
        probas.append(proba)

    probas = np.array(probas)
    cluster_labels = np.argmax(probas, axis=1)

    cluster_num = cluster_center_num
    cluster_labels_indices = [np.where(cluster_labels == i)[0] for i in np.arange(cluster_num)]

    image_labels = image_labels_test
    correct_labels = 0
    for index, indices in enumerate(cluster_labels_indices):
        image_label_cluster = image_labels[indices]
        image_label_percentage_cluster = [np.sum(image_label_cluster == i) / image_label_cluster.shape[0] for i in np.arange(cluster_num)]
        print('{}th clustering: '.format(index), image_label_percentage_cluster)
        most_labels = np.max(image_label_percentage_cluster) * image_label_cluster.shape[0]
        print('most_labels: ', most_labels)
        print('cluster size: ', image_label_cluster.shape[0])
        correct_labels += most_labels
    print('success rate: ', float(correct_labels) / len(image_labels))


def split_4class_2_11class(dataset_path: str, dataset_type: str):
    dataset_txt_path = os.path.join(dataset_path, 'file_list_{}.txt'.format(dataset_type))
    new_dataset_txt_path = os.path.join(dataset_path, 'file_list_state_{}.txt'.format(dataset_type))
    dataset_txt = open(dataset_txt_path, 'r')
    data = dataset_txt.read().split('\n')
    data = list(filter(lambda x: not x == '', data))

    seq_distribution = [0.2] * 5
    label_proj = {0: [0, 2, 4, 7, 10], 1: [1, 9], 3: [3, 6], 2: [5, 8]}
    label_seq_type = {0: 0,
                         1: 1, 2: 1,
                         3: 2, 4: 2,
                         5: 3, 6: 3, 7: 3,
                         8: 4, 9: 4, 10: 4}

    with open(new_dataset_txt_path, 'w') as f:

        for index in range(len(data)):
            img_path, label = data[index].split()
            potential_new_labels = label_proj[int(label)]
            new_label_distribution = [seq_distribution[label_seq_type[po_label]] for po_label in potential_new_labels]
            new_label_distribution = np.array(new_label_distribution) / np.linalg.norm(new_label_distribution, ord=1)
            new_label = np.random.choice(potential_new_labels, p=new_label_distribution)

            # if int(label) == 3:
            #     print('new_label: ', new_label)
            #     exit()

            f.writelines([img_path, ' ', str(new_label), '\n'])



if __name__ == '__main__':

    # root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/')
    # dataset_path = os.path.join(root_path, 'crop_dataset')
    # split_4class_2_11class(dataset_path=dataset_path, dataset_type='train')
    # split_4class_2_11class(dataset_path=dataset_path, dataset_type='test')
    # exit()
    # cluster_num()
    build_npy(encode_dataset=True, cluster=True)
    test_cluster_accuracy()