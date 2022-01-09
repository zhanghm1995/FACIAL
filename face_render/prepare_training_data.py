"""Created by Zhang Haiming
"""

import os, sys
import os.path as osp
import numpy as np
from scipy.io import loadmat,savemat
import glob
from scipy.signal import savgol_filter
import argparse
from load_data import BFM 
from face3d.morphable_model.fit import fit_points
from face3d import mesh


def parse_args():
    parser = argparse.ArgumentParser(description='netface_setting')
    parser.add_argument('--data_root', type=str, default='../video_preprocessed/id00001/gangqiang_3')
    opt = parser.parse_args()

    opt.dee3dface_param_folder = osp.join(opt.data_root, "deep3dface")
    opt.landmarks_path = osp.join(opt.data_root, "landmarks.npy")
    opt.save_path = osp.join(opt.data_root, "train_pose_new.npz")

    return opt


def get_coeff_vector(mat_path):
    """Get coefficient vector from Deep3DFace_Pytorch results

    Args:
        mat_path ([type]): [description]

    Returns:
        [type]: 1x257
    """
    keys_list = ['id', 'exp', 'tex', 'angle', 'gamma', 'trans']

    face_params = loadmat(mat_path)

    coeff_list = []
    for key in keys_list:
        coeff_list.append(face_params[key])
    
    coeff_res = np.concatenate(coeff_list, axis=1)
    return coeff_res


def get_deep3dface_matrix(opt, is_deep3dface_pytorch_version=True):
    param_folder = opt.dee3dface_param_folder

    mat_path_list = sorted(glob.glob(os.path.join(param_folder, '*.mat')))
    len_mat = len(mat_path_list)

    faceshape = np.zeros((len_mat, 257), float)

    if is_deep3dface_pytorch_version:
        for i, file in enumerate(mat_path_list):
            faceshape[i, :] = get_coeff_vector(file)[0]
    else:
        for i, file in enumerate(mat_path_list):
            faceshape[i, :] = loadmat(file)['coeff'][0, :]

    print(f"Deep3DFace matrix shape: {faceshape.shape}")
    return faceshape


def fit_headpose(opt, deep3dface_params):
    # --- 1. load BFM model
    facemodel = BFM()
    n_exp_para = facemodel.exBase.shape[1]

    kpt_ind = facemodel.keypoints
    

    ## --- 2. load 68 landmarks
    landmarks_npy_file = opt.landmarks_path
    all_frames_landmarsk = np.load(landmarks_npy_file)

    num_image = len(all_frames_landmarsk)

    realparams = deep3dface_params
    idparams = realparams[0, 0:80]
    texparams = realparams[0, 144:224]
    gammaparams = realparams[0, 227:254]

    h = 512
    w = 512

    headpose = np.zeros((num_image, 258), dtype=np.float32)

    # --- 2. fit head pose for each frame
    for frame_count in range(1, num_image+1):
        if frame_count % 1000 == 0:
            print(frame_count)
        
        curr_landmarks = all_frames_landmarsk[frame_count-1]
        
        x = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            x[i, 0] = curr_landmarks[i][0] - w/2
            x[i, 1] = (h - curr_landmarks[i][1]) - h/2 -1
        X_ind = kpt_ind

        fitted_sp, fitted_ep, fitted_s, fitted_R, fitted_t = fit_points(x, X_ind, facemodel, np.expand_dims(idparams,0), n_ep=n_exp_para, max_iter=10)

        fitted_angles = mesh.transform.matrix2angle(fitted_R)
        fitted_angles = np.array([fitted_angles])

        chi_prev = np.concatenate((fitted_angles[0,:], [fitted_s], fitted_t, realparams[frame_count-1, 80:144]), axis=0)
        params = np.concatenate((chi_prev, idparams, texparams, gammaparams), axis=0)
        headpose[frame_count-1, :] = params

    # additional smooth
    headpose1 = np.zeros((num_image, 258), dtype=np.float32)
    headpose1 = savgol_filter(headpose, 5, 3, axis=0)

    print(f"headpose shape: {headpose1.shape}")
    np.savez(opt.save_path, face=headpose1)



if __name__ == "__main__":
    opt = parse_args()

    deep3dface_matrix = get_deep3dface_matrix(opt)

    fit_headpose(opt, deep3dface_matrix)