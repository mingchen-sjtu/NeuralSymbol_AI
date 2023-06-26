#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2021

@author: DuYidong WangWenshuo
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


# build file_list (image_name, label)
def build_battery_dissemble_image_dataset(dataset_path):

    splits = ['train', 'test']
    split_ratio = 0.70

    labels = os.listdir(dataset_path)
    labels = list(filter(lambda x: os.path.isfile(os.path.join(dataset_path, x)) is False, labels))

    images_label_dict = {}
    for label in labels:
        label_images_path = os.path.join(dataset_path, label, 'rgb')
        label_images = os.listdir(label_images_path)
        images_label_dict[label] = label_images

    dataset_txt_path = os.path.join(dataset_path, 'file_list.txt')
    file_list = open(dataset_txt_path, 'w')

    split_file_list = {}
    for split in splits:
        split_file_txt_path = os.path.join(dataset_path, 'file_list_{}.txt'.format(split))
        split_file_temp = open(split_file_txt_path, 'w')
        split_file_list[split] = split_file_temp

    for label_index, label in enumerate(labels):

        label_split_file_list = {}
        for split in splits:
            label_split_file_txt_path = os.path.join(dataset_path, label, 'file_list_{}.txt'.format(split))
            label_split_file_temp = open(label_split_file_txt_path, 'w')
            label_split_file_list[split] = label_split_file_temp

        for image_name in images_label_dict[label]:

            split_rand_num = np.random.uniform()
            split = 'test' if bool(split_rand_num > split_ratio) else 'train'
            label_split_file_list[split].writelines([image_name, '\n'])

            split_file_list[split].writelines([label + '/rgb/' + image_name, ' ', str(label_index), '\n'])

            file_list.writelines([label + '/rgb/' + image_name, ' ', str(label_index), '\n'])

        for split in splits:
            label_split_file_list[split].close()

    for split in splits:
        split_file_list[split].close()

    file_list.close()


# build dataset_train/test (image_name_1, image_name_2, label)
def build_battery_dissemble_image_pair_dataset(dataset_path, dataset_size=5400):

    no_action_ratio = 0.35
    splits = ['train', 'test']
    split_ratio = {'train': 0.7, 'test': 0.3}
    images_num = {}
    stastic = {}

    labels = os.listdir(dataset_path)
    labels = list(filter(lambda x: os.path.isfile(os.path.join(dataset_path, x)) is False, labels))

    images_label_dict = {}
    for split in splits:
        images_label_dict[split] = {}
        for label in labels:
            label_images_path = os.path.join(dataset_path, label)
            # print('label_images_path: ', label_images_path)
            label_images = os.listdir(label_images_path)

            split_label_images_txt_path = os.path.join(dataset_path, label, 'file_list_{}.txt'.format(split))

            split_label_images_txt = open(split_label_images_txt_path, 'r')
            split_label_images = split_label_images_txt.read().split('\n')
            split_label_images = list(filter(lambda x: not x == '', split_label_images))
            
            images_label_dict[split][label] = split_label_images

            # summary images num in each cluster
            images_num[label] = len(os.listdir(label_images_path + '/rgb'))
            stastic[label] = 0
    # print('images_num: ', images_num)

    # hardcode images_num
    # images_num['no_clear_but_aim'] = 3400
    # images_num['no_clear_but_no_aim'] = 3400
    # exit()

    def FindUnseenData(label, all_data, seen_data):
        for _ in range(3):
            data = np.random.choice(all_data)
            if not label + '/' + data in seen_data:
                return data
        return data

    # calculate random choice weights
    def weights(images_num, labels):
        images_num = {label: images_num[label] for label in labels}
        images_num_values = np.asarray(list(images_num.values()))
        # print('images_num_values: ', images_num_values)
        images_sum = np.sum(images_num_values)
        weights = images_num_values / images_sum
        # print('weights: ', weights)
        return weights

    all_used_data = []
    weights = [0.1408, 0.3511, 0.3408, 0.103]
    weights = [float(i)/sum(weights) for i in weights]
    weights_dict = {'no_clear_but_no_aim': weights[0],
                    'clear_but_aim': weights[1],
                    'clear_but_no_aim': weights[2],
                    'no_clear_but_aim': weights[3]}
    # print('labels: ', labels)

    for split in splits:
        dataset_txt_path = os.path.join(dataset_path, 'dataset_{}.txt'.format(split))
        f = open(dataset_txt_path, 'w')

        for _ in range(int(np.ceil(split_ratio[split] * dataset_size))):
            # weighted random choice
            data_1_label = np.random.choice(labels, 1, p=weights)[0]
            # data_1_label = np.random.choice(labels, 1, p=weights(images_num, labels))[0]

            # data_1 = np.random.choice(images_label_dict[split][data_1_label])
            data_1 = FindUnseenData(label=data_1_label, all_data=images_label_dict[split][data_1_label], seen_data=all_used_data)   

            rand_action_sample = np.random.uniform()
            action = bool(rand_action_sample > no_action_ratio)

            if action:
                new_dict = {key:val for key, val in weights_dict.items() if key != data_1_label}
                weights_2 = list(new_dict.values())
                weights_2 = [float(i) / sum(weights_2) for i in weights_2]
                data_2_label = np.random.choice(list(set(labels) - {data_1_label}), 1, p=weights_2)[0]
                # data_2_label = np.random.choice(list(set(labels) - {data_1_label}), 1, p=weights(images_num, list(set(labels) - {data_1_label})))[0]

                while data_1_label == 'clear_but_no_aim' and data_2_label == 'no_clear_but_aim' or\
                    data_2_label == 'clear_but_no_aim' and data_1_label == 'no_clear_but_aim':
                    data_2_label = np.random.choice(list(set(labels) - {data_1_label}), 1, p=weights_2)[0]

            else:
                data_2_label = data_1_label
                # data_2 = np.random.choice(images_label_dict[split][data_2_label])

            data_2 = FindUnseenData(label=data_2_label, all_data=images_label_dict[split][data_2_label], seen_data=all_used_data)
            all_used_data.append(data_1_label + '/' + data_1)
            all_used_data.append(data_2_label + '/' + data_2)

            action_label = int(action)
            if data_1_label == 'clear_but_aim' and data_2_label == 'no_clear_but_no_aim' or\
                    data_2_label == 'clear_but_aim' and data_1_label == 'no_clear_but_no_aim':
                action_label = 2
            f.writelines([data_1_label + '/rgb/' + data_1, ' ', data_2_label + '/rgb/' + data_2, ' ', str(action_label), '\n'])

            stastic[data_1_label] += 1
            stastic[data_2_label] += 1
        f.close()

    # print('stastic: ', stastic)
    # print('images_num: ', images_num)
    return stastic, images_num


# build action_sequence_train/test
def build_action_sequence(dataset_path, dataset_size=1000):
# Parameter:
#           dataset_path, dataset_size: sum sequence length in train/text txt
# Return:
#           dataset_train(test)_from_seq.txt: e.g.(approach insert rgb_img_xxx.png clear_but_aim )

    primative_type = ['approach', 'mate', 'push', 'insert']
    state_type = ['clear_but_aim', 'clear_but_no_aim', 'no_clear_but_aim', 'no_clear_but_no_aim']

    action_types = [ 'approach_insert',
                     'approach_push_insert',
                     'approach_mate_insert',
                     'approach_push_mate_insert']

    action_types_labels = { 'approach_insert': [1],
                            'approach_push_insert': [2, 3],
                            'approach_mate_insert': [4, 5],
                            'approach_push_mate_insert': [6, 7, 8]
                            }

    class_types_labels = {  1: 'clear_but_aim', 2: 'no_clear_but_aim', 3: 'clear_but_aim',
                            4: 'clear_but_no_aim', 5: 'clear_but_aim', 6: 'no_clear_but_no_aim',
                            7: 'clear_but_no_aim', 8: 'clear_but_aim'}

    action_sequence_ratio = [0.1, 0.3, 0.3, 0.3]
    splits = ['train', 'test']
    split_ratio = {'train': 0.7, 'test': 0.3}

    # labels: ['no_clear_but_no_aim', 'clear_but_aim', 'clear_but_no_aim', 'no_clear_but_aim']
    labels = os.listdir(dataset_path)
    labels = list(filter(lambda x: os.path.isfile(os.path.join(dataset_path, x)) is False, labels))


    # dict[train/test][labels] = [images names]
    images_label_dict = {}
    for split in splits:
        images_label_dict[split] = {}
        for label in labels:
            label_images_path = os.path.join(dataset_path, label)
            label_images = os.listdir(label_images_path)

            split_label_images_txt_path = os.path.join(dataset_path, label, 'file_list_{}.txt'.format(split))

            split_label_images_txt = open(split_label_images_txt_path, 'r')
            split_label_images = split_label_images_txt.read().split('\n')
            split_label_images = list(filter(lambda x: not x == '', split_label_images))
            
            images_label_dict[split][label] = split_label_images

    # build action_sequence_train & action_sequence_test txt
    all_used_data = []
    for split in splits:

        dataset_txt_path = os.path.join(dataset_path, 'action_sequence_{}.txt'.format(split))
        f = open(dataset_txt_path, 'w')

        sequence_len = 0
        while sequence_len < np.ceil(dataset_size * split_ratio[split]):
            writing_line = []
            ground_truth = []

            # action term
            rand_sequence = np.random.choice(action_types, p=action_sequence_ratio)
            actions = rand_sequence.split('_')
            for act in actions:
                writing_line.extend((act, ' '))

            # image names term
            rand_seq_states = action_types_labels[rand_sequence]
            for rand_seq_state in rand_seq_states:
                state_label = class_types_labels[rand_seq_state]
                all_data = images_label_dict[split][state_label]
                state_data = np.random.choice(all_data)
                # all_used_data.extend(all_data)
                # state_data = FindUnseenData(label=state_label, all_data=images_label_dict[split][state_label], seen_data=all_used_data)
                writing_line.extend((state_data, ' '))

                ground_truth.append(state_label)

            # ground truth term
            for ground in ground_truth:
                writing_line.extend((ground, ' '))

            writing_line.extend('\n')
            # print('line: ', writing_line)

            f.writelines(writing_line)
            sequence_len += 1

        f.close()


# build repelling sets from action sequence
def build_repell_appeal_set():

    action_types_labels = { 'approach_insert': [1],
                            'approach_push_insert': [2, 3],
                            'approach_mate_insert': [4, 5],
                            'approach_push_mate_insert': [6, 7, 8]}

    # build appeal sets
    appeal_sets = [[1, 3, 5, 8], [4, 7]]
    appeal_set = []
    for values in appeal_sets:
        for i in range(len(values) - 1):
            for j in range(i+1, len(values)):
                appeal_set.append(set([values[i], values[j]]))

    # hardcode add appeal sets
    appeal_set.append([2])
    appeal_set.append([4])
    appeal_set.append([6])
    appeal_set.append([7])

    # build repell sets
    repell_set = []
    for key in list(action_types_labels.keys()):
        values = action_types_labels[key]
        if len(values) > 1:
            for i in range(len(values) - 1):
                for j in range(i + 1, len(values)):
                    repell_set.append(set([values[i], values[j]]))

    # build "non-related" sets following linking rules
    non_related_set = []
    for i in range(1, 8):
        for j in range(i + 1, 8+1):
            set_i_j = set([i, j])

            # first check if [i, j] is in built appeal&repell sets
            if set_i_j in appeal_set or set_i_j in repell_set:
                continue

            flag = 0
            cache_list = [x for x in list(range(1, 8 + 1)) if not x in set_i_j]
            for m in cache_list:
                # Add repell sets following linking rules
                if (set([i, m]) in appeal_set and set([j, m]) in repell_set) \
                    or (set([i, m]) in repell_set and set([j, m]) in appeal_set):
                    repell_set.append(set_i_j)
                    flag = 1
                    break

            if flag == 0:
                non_related_set.append(set_i_j)

    # print('appeal_set: ', appeal_set)
    # print('repell_set: ', repell_set)
    # print('non_related_set: ', non_related_set)
    # exit()

    return appeal_set, repell_set, non_related_set
 

# read and split the information from action sequence txt file
def read_sequence_txt(txt_file_path: str):
# Parameter:
#           txt_file_path: path of the txt file, str
# Return:
#           num_sequences:  action sequence as integers, list
#           images_paths:   image paths, list
#           GT_state_nums:  groundtruth states, list

    action_num_dict = {'approach': 0, 'insert': 1, 'push': 2, 'mate': 3}
    state_num_dict = {'no_clear_but_no_aim': 0, 'no_clear_but_aim': 1, 'clear_but_no_aim': 2, 'clear_but_aim': 3}
    num_sequences = []
    images_paths = []
    GT_state_nums = []

    with open(txt_file_path, 'r') as seq_txt_file:
        sequences = seq_txt_file.read().split('\n')
        sequences = list(filter(lambda x: not x == '', sequences))
        for sequence in sequences:
            data = sequence.split(' ')
            data = list(filter(lambda x: not x == '', data))

            num_sequence = [action_num_dict[x] for x in data if x in list(action_num_dict.keys())]
            GT_state_num = [state_num_dict[x] for x in data if x in list(state_num_dict.keys())]
            image_path = [x for x in data if not x in list(state_num_dict.keys()) and not x in list(action_num_dict.keys())]

            num_sequences.append(num_sequence)
            images_paths.append(image_path)
            GT_state_nums.append(GT_state_num)

    # print('num_sequences: ', num_sequences)
    # print('GT_state_nums: ', GT_state_nums)
    # build_supervised_learning_dataset(num_sequences, images_paths, GT_state_nums)

    return num_sequences, images_paths, GT_state_nums


# build dataset from repell and appeal sets
def build_dataset_from_repell_appeal_sets(dataset_path, dataset_size):
# Parameter:
#           dataset_path, dataset_size
# Return:
#           dataset_train(test)_from_seq.txt: images pair + action label(0, 1, 2)

    class_types_labels = {  1: 'clear_but_aim', 2: 'no_clear_but_aim', 3: 'clear_but_aim',
                            4: 'clear_but_no_aim', 5: 'clear_but_aim', 6: 'no_clear_but_no_aim',
                            7: 'clear_but_no_aim', 8: 'clear_but_aim'}

    appeal_set, repell_set, non_related_set = build_repell_appeal_set()
    sets = [appeal_set, repell_set, non_related_set]

    # label 0/1/2 ratio
    label_ratio = [1/3, 1/3, 1/3]
    splits = ['train', 'test']
    split_ratio = {'train': 0.7, 'test': 0.3}

    for split in splits:
        dataset_txt_path = os.path.join(dataset_path, 'dataset_{}.txt'.format(split))
        f = open(dataset_txt_path, 'w')

        # write label 0/1/2 data in file
        data_size = 0
        for i in range(len(label_ratio)):
            while data_size < np.ceil(dataset_size * split_ratio[split] * label_ratio[i]):
                select_set = random.choice(sets[i])
                if len(list(select_set)) == 1:
                    class_label_0 = class_types_labels[list(select_set)[0]]
                    class_label_1 = class_types_labels[list(select_set)[0]]
                else:
                    class_label_0 = class_types_labels[list(select_set)[0]]
                    class_label_1 = class_types_labels[list(select_set)[1]]

                rgb_img_0 = random.choice(os.listdir(os.path.join(dataset_path, class_label_0, 'rgb')))
                rgb_img_1 = random.choice(os.listdir(os.path.join(dataset_path, class_label_1, 'rgb')))

                path_0 = class_label_0 + '/rgb/' + rgb_img_0
                path_1 = class_label_1 + '/rgb/' + rgb_img_1

                f.writelines([path_0, ' ', path_1, ' ', str(i), '\n'])

                data_size += 1
            data_size = 0


# unsupervised learning, same subsequent actions rules(class_num: 4)
def build_dataset_from_action_seq_class_4(dataset_path):
# Parameter:
#           dataset_path
# Return:
#           dataset_train(test)_from_seq.txt: images pair + action label(0, 1, 2)

    # label 1/2 : label 0 = 0.65 : 0.35
    action_ratio = 0.65
    no_action_2_action_ratio = (1 - action_ratio) / action_ratio
    no_action_weight = [0.25, 0.25, 0.25, 0.25]
    splits = ['train', 'test']
    GT_state_dict = {'0': 'no_clear_but_no_aim', '1': 'no_clear_but_aim', '2': 'clear_but_no_aim', '3': 'clear_but_aim'}

    for split in splits:
        # call read_sequence_txt fucntion
        file_path = os.path.join(dataset_path, 'action_sequence_{}.txt'.format(split))
        num_sequences, images_paths, GT_state_nums = read_sequence_txt(txt_file_path=file_path)

        dataset_txt_path = os.path.join(dataset_path, 'dataset_{}.txt'.format(split))
        f = open(dataset_txt_path, 'w')

        # write label 2 data in file
        action_2_len = 0
        for index, images_path in enumerate(images_paths):
            if len(images_path) == 2:

                path_0 = GT_state_dict[str(GT_state_nums[index][0])] + '/rgb/' + images_path[0]
                path_1 = GT_state_dict[str(GT_state_nums[index][1])] + '/rgb/' + images_path[1]

                f.writelines([path_0, ' ', path_1, ' ', str(2), '\n'])
                action_2_len += 1

            if len(images_path) == 3:

                path_0 = GT_state_dict[str(GT_state_nums[index][0])] + '/rgb/' + images_path[0]
                path_1 = GT_state_dict[str(GT_state_nums[index][1])] + '/rgb/' + images_path[1]
                path_2 = GT_state_dict[str(GT_state_nums[index][2])] + '/rgb/' + images_path[2]

                f.writelines([path_0, ' ', path_1, ' ', str(2), '\n'])
                f.writelines([path_1, ' ', path_2, ' ', str(2), '\n'])
                f.writelines([path_0, ' ', path_2, ' ', str(2), '\n'])
                action_2_len += 3

        # write label 1(no-related) data in file
        action_1_len = 0
        while action_1_len < np.ceil(action_2_len / 2):
            name_1 = random.choice(os.listdir(os.path.join(dataset_path, 'clear_but_no_aim', 'rgb')))
            name_2 = random.choice(os.listdir(os.path.join(dataset_path, 'no_clear_but_aim', 'rgb')))
            path_1 = 'clear_but_no_aim/rgb/' + name_1
            path_2 = 'no_clear_but_aim/rgb/' + name_2
            f.writelines([path_1, ' ', path_2, ' ', str(1), '\n'])

            name_2 = random.choice(os.listdir(os.path.join(dataset_path, 'no_clear_but_aim', 'rgb')))
            name_3 = random.choice(os.listdir(os.path.join(dataset_path, 'no_clear_but_no_aim', 'rgb')))
            path_2 = 'no_clear_but_aim/rgb/' + name_2
            path_3 = 'no_clear_but_no_aim/rgb/' + name_3
            f.writelines([path_2, ' ', path_3, ' ', str(1), '\n'])
            action_1_len += 2

        # build unique states sets obeying "all states in each set that share same subsequent actions"
        appeared_sub_act = set()
        for sequence in num_sequences:
            for i in range(1, len(sequence)):
                sub_act_str = '_'.join([str(x) for x in sequence[i:]])
                appeared_sub_act.add(sub_act_str)
        # print('appeared_sub_act: ', appeared_sub_act)

        # build dict sets
        no_action_dict = {}
        for unique_item in appeared_sub_act:
            no_action_dict[unique_item] = []
        # print('no_action_dict: ', no_action_dict)

        # loop action sequences to add images path in dict sets
        for index, sequence in enumerate(num_sequences):
            for i in range(1, len(sequence)):
                sub_act_str = '_'.join([str(x) for x in sequence[i:]])
                path = GT_state_dict[str(GT_state_nums[index][i-1])] + '/rgb/' + images_paths[index][i-1]
                no_action_dict[sub_act_str].append(path)
        # print('no_action_dict: ', no_action_dict)

        # write label 0 data in file
        no_action_len = 0
        for _ in range(int((action_1_len + action_2_len) * no_action_2_action_ratio)):
            dict_key = random.choice(list(no_action_dict.keys()))
            # dict_key = np.random.choice(list(no_action_dict.keys()), 1, p=no_action_weight)[0]
            # print('dict_key: ', dict_key)
            image_pair = random.choices(no_action_dict[dict_key], k=2)
            f.writelines([image_pair[0], ' ', image_pair[1], ' ', str(0), '\n'])
            no_action_len += 1

        f.close()


# unsupervised learning, same subsequent actions rules(class_num: 4)
def build_dataset_from_action_seq_class_11(dataset_path):
# Parameter:
#           dataset_path
# Return:
#           dataset_train(test)_from_seq.txt: images pair + action label(0, 1, 2)

    # label 1/2 : label 0 = 0.65 : 0.35
    action_ratio = 0.65
    no_action_2_action_ratio = (1 - action_ratio) / action_ratio
    splits = ['train', 'test']
    GT_state_dict = {'0': 'no_clear_but_no_aim', '1': 'no_clear_but_aim', '2': 'clear_but_no_aim', '3': 'clear_but_aim'}

    for split in splits:
        # call read_sequence_txt fucntion
        file_path = os.path.join(dataset_path, 'action_sequence_{}.txt'.format(split))
        num_sequences, images_paths, GT_state_nums = read_sequence_txt(txt_file_path=file_path)

        dataset_txt_path = os.path.join(dataset_path, 'dataset_{}.txt'.format(split))
        f = open(dataset_txt_path, 'w')

        # write label 1/2 data in file
        action_len = 0
        for index, images_path in enumerate(images_paths):
            if len(images_path) == 2:

                path_0 = GT_state_dict[str(GT_state_nums[index][0])] + '/rgb/' + images_path[0]
                path_1 = GT_state_dict[str(GT_state_nums[index][1])] + '/rgb/' + images_path[1]

                f.writelines([path_0, ' ', path_1, ' ', str(1), '\n'])
                action_len += 1

            if len(images_path) == 3:

                path_0 = GT_state_dict[str(GT_state_nums[index][0])] + '/rgb/' + images_path[0]
                path_1 = GT_state_dict[str(GT_state_nums[index][1])] + '/rgb/' + images_path[1]
                path_2 = GT_state_dict[str(GT_state_nums[index][2])] + '/rgb/' + images_path[2]

                f.writelines([path_0, ' ', path_1, ' ', str(1), '\n'])
                f.writelines([path_1, ' ', path_2, ' ', str(1), '\n'])
                f.writelines([path_0, ' ', path_2, ' ', str(2), '\n'])
                action_len += 3

        # build unique states sets obeying "all states in each set that share same subsequent actions"
        appeared_act_seq = set()
        for sequence in num_sequences:
            for i in range(len(sequence)):
                act_seq_str = '_'.join([str(x) for x in sequence])
                appeared_act_seq.add(act_seq_str)
        # print('appeared_act_seq: ', appeared_act_seq)


        # build dict sets
        no_action_dict = {}
        for unique_item in appeared_act_seq:
            no_action_dict[unique_item] = []
        # print('no_action_dict: ', no_action_dict)

        # loop action sequences to add images path in dict sets
        for index, sequence in enumerate(num_sequences):
            act_seq_str = '_'.join([str(x) for x in sequence])

            paths = []
            GT_str = [GT_state_dict[str(x)] for x in GT_state_nums[index]]
            for i in range(len(GT_str)):
                path = GT_str[i] + '/rgb/' + images_paths[index][i]
                paths.append(path)
            
            no_action_dict[act_seq_str].append(paths)

        # write label 0 data in file
        no_action_len = 0
        for _ in range(int(action_len * no_action_2_action_ratio)):
            dict_key = random.choice(list(no_action_dict.keys()))
            # print('dict_key: ', dict_key)

            row_id = np.asarray(no_action_dict[dict_key]).shape[0]
            col_id = np.asarray(no_action_dict[dict_key]).shape[1]
            random_row_id = np.random.choice(row_id, 2)
            random_col_id = np.random.choice(col_id, 1)

            image_pair = np.asarray(no_action_dict[dict_key])[random_row_id, random_col_id]

            f.writelines([image_pair[0], ' ', image_pair[1], ' ', str(0), '\n'])
            no_action_len += 1

        print('action_len: ', action_len)
        print('no_action_len: ', no_action_len)
        f.close()


# build dataset for supervised learning
def build_supervised_learning_dataset(dataset_path, dataset_size=20):

    primative_type = ['approach', 'mate', 'push', 'insert']
    splits = ['train', 'test']
    split_ratio = {'train': 0.7, 'test': 0.3}
    choice_ratio = [0.2] * 5

    action_types_labels = {'approach_insert': [1],
                           'approach_mate_insert': [2, 3],
                           'approach_push_insert': [4, 5],
                           'approach_push_mate_insert': [6, 7, 8],
                           'approach_mate_push_insert': [9, 10, 11]}

    for split in splits:
        txt_file_path = os.path.join(dataset_path, 'action_sequence_{}.txt'.format(split))
        with open(txt_file_path, 'r') as seq_txt_file:
            sequences = seq_txt_file.read().split('\n')

        supervised_dataset_path = os.path.join(dataset_path, 'supervised_dataset_{}.txt'.format(split))
        f = open(supervised_dataset_path, 'w')

        collected_dataset_size = 0
        while collected_dataset_size < np.ceil(dataset_size * split_ratio[split]):
            actions = []
            sequence = np.random.choice(sequences)
            sequence = sequence.split(' ')
            print('sequence: ', sequence)

            for action in sequence:
                if action in primative_type:
                    print('action: ', action)
                    actions.append(action)
            if len(actions) == 0:
                continue

            print('actions: ', actions)
            action_sequence = '_'.join(actions)
            print('action_sequence: ', action_sequence)

            supervised_labels = action_types_labels[action_sequence]
            print('supervised_label: ', supervised_labels)
            for supervised_label_index, supervised_label in enumerate(supervised_labels):
                writing_line = []
                writing_line.extend((sequence[len(supervised_labels) + supervised_label_index + 1], ' ', str(supervised_label), '\n'))
                print('writing_line: ', writing_line)
                f.writelines(writing_line)

            collected_dataset_size += len(supervised_labels)
        f.close()


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_path, 'crop_dataset')

    # build_action_sequence(data_path, dataset_size=2257)
    # build_dataset_from_action_seq_class_4(data_path)

    build_dataset_from_repell_appeal_sets(data_path, dataset_size=4000)
