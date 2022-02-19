'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-19 11:32:34
Email: haimingzhang@link.cuhk.edu.cn
Description: Combine the deep3dface reconstruction results to a single matrix
'''

import os, sys
import numpy as np
from scipy.io import loadmat,savemat
import glob
from scipy.signal import savgol_filter
import argparse

parser = argparse.ArgumentParser(description='netface_setting')
parser.add_argument('--param_folder', type=str, default='../video_preprocess/deep3dface')

opt = parser.parse_args()

param_folder = opt.param_folder

mat_path_list = sorted(glob.glob(os.path.join(param_folder, '*.mat')))
len_mat = len(mat_path_list)
faceshape = np.zeros((len_mat, 257), float)

for i, file in enumerate(mat_path_list):
    faceshape[i, :] = loadmat(file)['coeff'][0, :]

print(f"Deep3DFace matrix shape: {faceshape.shape}")
frames_out_path = os.path.join(param_folder,'deep3dface.npz')
np.savez(frames_out_path, face=faceshape) # (num_frame, 257)