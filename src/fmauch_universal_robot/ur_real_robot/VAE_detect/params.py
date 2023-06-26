import sys
import os
import torch
import numpy as np

root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
dataset_path = os.path.join(root_path, 'crop_dataset')
npy_path = os.path.join(root_path, 'npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_file = "VAE_ShirtFolding_L1"
checkpoint_file = "vae_best_checkpoint.pth"
# checkpoint_file = "vae_lastCheckpoint.pth"
img_channels_num = 3
batch_size = 16

encoded_latent_vectors_npy = 'npy/latent_vectors.npy'
image_labels_npy = 'npy/image_labels.npy'

cluster_means_npy = 'npy/numpy_means.npy'
cluster_covs_npy = 'npy/numpy_covs.npy'

label_nums_npy = 'npy/label_nums.npy'
end_state_distribution_npy = 'npy/end_state_distribution.npy'

cluster_center_num = 4
label_num = 11
action_num = 3
search_level = 3
search_order = ['Push', 'Mate', "Self"]

lower_KL_divergance_threshold = 0.4
higher_KL_divergance_threshold = 10.0

Push = [[1, 2], [5, 6], [9, 10]]
Mate = [[3, 4], [6, 7], [8, 9]]
Self = [[0, 0], [2, 2], [4, 4], [7, 7], [10, 10]]

Push_self = [[0, 0], [2, 2], [3, 3], [4, 4], [6, 6], [7, 7], [8, 8], [10, 10]]
Mate_self = [[0, 0], [1, 1], [2, 2], [4, 4], [5, 5], [7, 7], [9, 9], [10, 10]]

action_type = {0: 'clear_aim', 1: 'no_clear_aim', 2: 'no_clear_no_aim', 3: 'clear_no_aim'}

Push_dynamic = np.array([[1, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 1]])
Mate_dynamic = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 1, 0, 0],
                         [1, 0, 0, 0]])
Self_dynamic = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

end_states = [0, 2, 4, 7, 10]
end_label = [0]

clear_aim_states = [0, 2, 4, 7, 10]
no_clear_aim_states = [1, 9]
no_clear_no_aim_states = [5, 8]
clear_no_aim_states = [3, 6]

clear_aim_label = [0]
no_clear_aim_label = [1]
no_clear_no_aim_label = [2]
clear_no_aim_label = [3]

groundtruth_plan = {'clear_aim': [['End']],
					'no_clear_aim': [['Push', "End"]],
					'no_clear_no_aim': [['Mate', 'Push', "End"], ['Push', 'Mate', "End"]],
					'clear_no_aim': [['Mate', "End"]]}

# goal proba:  [0.06775339 0.8936382  0.00267348 0.03593494]
# goal proba:  [0.00423074 0.0454454  0.05886511 0.89145875]
# goal proba:  [0.04979726 0.00600547 0.82074177 0.1234555 ]
# goal proba:  [0.82027787 0.08784656 0.07454447 0.01733111]

# cluster size:  2757
# cluster size:  1651
# cluster size:  1140
# cluster size:  812

# cur_latent_vectors:  (1341, 64)
# cur_latent_vectors:  (2576, 64)
# cur_latent_vectors:  (2228, 64)
# cur_latent_vectors:  (1183, 64)