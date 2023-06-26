#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2022

@author: DuYidong WangWenshuo
"""

import sys
import os
import numpy as np
root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(root_path)

from params import *
from cluster import *


if __name__ == '__main__':

	vae_model = load_vae_model()

	# build class_4
	build_npy(model=vae_model, encode_dataset=True, cluster=True, class_nums=4)

	# build class_11
	build_npy(model=vae_model, encode_dataset=True, cluster=True, class_nums=11)
