import sys
import os
import numpy as np

from importlib.machinery import SourceFileLoader
import algorithms as alg
from dataloader import BatteryDissembleImageDataset
import torch
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize

from params import *


def file_path(file_name: str, file_path: bool=True, split: str='train'):
	file_name = os.path.join(root_path, file_name)
	file_name, file_format = file_name.split('.')
	if not split is None:
		file_name = file_name + '_' + split
	if file_path:
		return file_name + '.' + file_format
	else:
		return file_name


# load VAE model
def load_vae_model():

	vae_config_file = os.path.join('.', 'configs', config_file + '.py')
	vae_directory = os.path.join('.', 'models', checkpoint_file)
	vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config

	vae_config['exp_name'] = config_file
	vae_config['vae_opt']['exp_dir'] = vae_directory
	vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
	vae_algorithm.load_checkpoint('models/' + config_file + "/" + checkpoint_file)
	vae_algorithm.model.eval()

	return vae_algorithm.model


# train/test dataloader
def enocde_dataset(model, dataset_type, class_nums):
	# test_dataloader
	root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/')
	dataset_path = os.path.join(root_path, 'crop_dataset')
	# print('dataset_path: ', dataset_path)
	battery_set = BatteryDissembleImageDataset(dataset_path=dataset_path, dataset_type=dataset_type, class_nums=class_nums)

	dataloader = torch.utils.data.DataLoader(battery_set, batch_size=batch_size, shuffle=True,
	        num_workers=0, drop_last=True)

	# image_labels & latent_vectors
	image_labels = np.zeros((0))
	latent_vectors = np.zeros((0, 64))
	for batch_idx, (img, image_label) in enumerate(dataloader):
	    img = img.to(device)
	    image_labels = np.append(image_labels, image_label.cpu().detach().numpy())
	    # print('image_labels: ', image_labels)

	    z_img = model(img, sample_latent=True, latent_code=True)
	    latent_vectors = np.append(latent_vectors, z_img.cpu().detach().numpy(), axis=0)
	print('image_labels: ', image_labels)
	return latent_vectors, image_labels


def latent_vec_2_cluster_proba(latent_vector: np.ndarray, numpy_means=None, numpy_covs=None):
	if numpy_means is None or numpy_covs is None:
	    numpy_means = np.load(file_path(file_name=cluster_means_npy, file_path=True, split=None))
	    numpy_covs = np.load(file_path(file_name=cluster_covs_npy, file_path=True, split=None))

	probas = []
	for i in range(numpy_means.shape[0]):
	    mean = numpy_means[i, :]
	    cov = numpy_covs[i, :, :]
	    proba = np.array(multivariate_normal.pdf(latent_vector, mean=mean, cov=cov))
	    probas.append(proba)
	probas = np.array(probas).reshape(1, -1)

	probas_norm = normalize(probas, norm='l1')[0]
	return probas_norm


def eval_data_distribution(image_labels_test, latent_vectors_test):

	print(np.unique(image_labels_test))
	vector_index_options = np.where(image_labels_test == 0)[0]
	print('cluster size: ', vector_index_options.shape)
	summation = np.zeros((cluster_center_num))
	for index in vector_index_options:
	    summation = latent_vec_2_cluster_proba(latent_vectors_test[index, :]) + summation
	    # print('latent_vec_2_cluster_proba(latent_vectors_test[index, :]): ', np.round(latent_vec_2_cluster_proba(latent_vectors_test[index, :]), 3))

	print('goal proba: ', summation / np.sum(summation))

	vector_index_options = np.where(image_labels_test == 1)[0]
	print('cluster size: ', vector_index_options.shape)
	summation = np.zeros((cluster_center_num))
	for index in vector_index_options:
	    summation = latent_vec_2_cluster_proba(latent_vectors_test[index, :]) + summation

	print('goal proba: ', summation / np.sum(summation))

	vector_index_options = np.where(image_labels_test == 2)[0]
	print('cluster size: ', vector_index_options.shape)
	summation = np.zeros((cluster_center_num))
	for index in vector_index_options:
	    summation = latent_vec_2_cluster_proba(latent_vectors_test[index, :]) + summation
	    # print('latent_vec_2_cluster_proba(latent_vectors_test[index, :]): ', np.round(latent_vec_2_cluster_proba(latent_vectors_test[index, :]), 3))
	print('goal proba: ', summation / np.sum(summation))
	# exit()

	vector_index_options = np.where(image_labels_test == 3)[0]
	print('cluster size: ', vector_index_options.shape)
	summation = np.zeros((cluster_center_num))
	for index in vector_index_options:
	    summation = latent_vec_2_cluster_proba(latent_vectors_test[index, :]) + summation

	print('goal proba: ', summation / np.sum(summation))
	return