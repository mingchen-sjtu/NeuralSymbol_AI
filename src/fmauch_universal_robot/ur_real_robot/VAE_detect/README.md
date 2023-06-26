# Intel_TAMP

## Table of contents
* [General info](#general-info)
* [Environments](#environments)
* [Setup](#setup)
* [Train VAE Model](#train-vae-model)

### General info
NeuroSymbolic TAMP for Battery Disassembly Task

---
### Environments
Project is created with:
```
* Ubuntu Release: 20.10
* Nvidia Driver: 470.57.02
* CUDA Version: 11.4
* Python: 3.8.10
```
---


### Setup
To run this project, install it locally using git and pip:
```
$ git clone https://gitee.com/du-yidong/intel_-tamp.git
$ cd ../intel_-tamp
$ pip install -r requirements.txt
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
---

### Images Dataset (sim & real)
```
https://pan.baidu.com/s/1BBRkZ13bCbWDWy4iZML-YA?pwd=data
```
---

### Train VAE Model
Train VAE model initially
```
$ python train_VAE.py --exp_vae=VAE_ShirtFolding_L1 --cuda=True
```

Train VAE model from last checkpoint
```
$ python train_VAE.py --exp_vae=VAE_ShirtFolding_L1 --cuda=True --chpnt_path ./models/VAE_ShirtFolding_L1/vae_lastCheckpoint.pth
```
---

### Cluster
Generate the probabilities of clusters given an image path
```
$ python demo.py
```
---

### Planner
1. Build the npy files, including:

image_labels_train.npy, image_labels_test.npy ; latent_vectors_train.npy, latent_vectors_test.npy, numpy_means.npy, numpy_covs.npy, label_nums.npy
```
python build_plan_npy.py
```
2. Plan
```
python plan.py
```
---

### Transfer STP to Neural Network
Put the encoded 'labels.npy' and 'vectors.npy' files(by running img2vec.py) into stp2nn folder
```
python train_vec2vec.py
```
---

### VAE decoding and reconstruct Images
```
python reconstruct_plot.py
```
