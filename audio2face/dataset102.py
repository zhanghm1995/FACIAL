import os
import glob
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class Facial_Dataset(Dataset):
    def __init__(self, audio_paths, npz_paths, cvs_paths=None):
        """[summary]

        Args:
            audio_paths ([list]): Audio feature pickle file path list
            npz_paths (list): Face 3D parameters npz file path list
            cvs_paths (list, optional): Face 3D Head pose csv file path list. Defaults to None.
        """
        self.audio_path_list = audio_paths
        self.mesh_param_path_list = npz_paths
        self.blink_path_list = cvs_paths if cvs_paths is not None else [None] * len(npz_paths)

        self.frames = 128

        self.dataset_audio = []
        self.dataset_exp_param = []
        self.dataset_idx = []
        base = 0

        for audio_path, param_path, blink_path in zip(self.audio_path_list, self.mesh_param_path_list, self.blink_path_list):
            audio_name = audio_path.split('/')[-1].replace('.pkl', '')
            param_name = param_path.split('/')[-1].replace('.npz', '')
            try:
                assert audio_name == param_name
            except:
                print(audio_name, param_name)
                
            audio = pickle.load(open(audio_path, 'rb'), encoding=' iso-8859-1')
            params = np.load(open(param_path, 'rb'))['face']

            if blink_path is not None:
                blinkinfo = pd.read_csv(blink_path)
                try:
                    aublink = blinkinfo['AU45_r'].values
                except:
                    aublink = blinkinfo[' AU45_r'].values
            else:
                aublink = np.zeros((audio.shape[0],))

            std1 = np.std(params, axis=0)
            mean1 = np.mean(params,axis=0)

            for i in range(6):
                params[:,i] = (params[:,i] - mean1[i]) / std1[i]
            
            min_num_frame = min(params.shape[0], audio.shape[0])

            if params.shape[0] != audio.shape[0]:
                params = params[0:min_num_frame,:]
                audio = audio[0:min_num_frame,:,:]
                aublink = aublink[0:min_num_frame]

            aublink = aublink[:, np.newaxis]

            self.dataset_audio.append(audio)
            self.dataset_exp_param.append(np.concatenate((params[:,:6], aublink, params[:,7:71]), axis=1))
            self.dataset_idx += list(np.arange(0, min_num_frame-self.frames, 1) + base)
            base+= min_num_frame


        self.dataset_audio = torch.Tensor(np.concatenate(self.dataset_audio))
        self.dataset_exp_param = torch.Tensor(np.concatenate(self.dataset_exp_param))
        self.dataset_idx = torch.Tensor(np.array(self.dataset_idx)).to(torch.long)
        print(max(self.dataset_idx))
        assert self.dataset_audio.shape[0] == self.dataset_exp_param.shape[0] 


    def __len__(self):
        return len(self.dataset_idx)

    def __getitem__(self, idx):

        audio = self.dataset_audio[self.dataset_idx[idx]:self.dataset_idx[idx]+self.frames]

        return audio, self.dataset_exp_param[self.dataset_idx[idx]:self.dataset_idx[idx]+self.frames]


class ExpressionDataset(Dataset):
    """Only consider the expression parameters
    """
    def __init__(self, audio_paths, deep3dface_param_paths):
        """[summary]

        Args:
            audio_paths ([list]): Audio feature pickle file path list
            deep3dface_param_paths (list): FaceRecon 3D parameters file path list
        """
        self.audio_path_list = audio_paths
        self.mesh_param_path_list = deep3dface_param_paths

        self.frames = 128

        self.dataset_audio = []
        self.dataset_exp_param = []
        self.dataset_idx = []
        
        base = 0

        for audio_path, param_path in zip(self.audio_path_list, self.mesh_param_path_list):
            audio = pickle.load(open(audio_path, 'rb'), encoding=' iso-8859-1')
            params = np.load(open(param_path, 'rb'))['face']

            min_num_frame = min(params.shape[0], audio.shape[0])

            if params.shape[0] != audio.shape[0]:
                params = params[0:min_num_frame,:]
                audio = audio[0:min_num_frame,:,:]

            self.dataset_audio.append(audio)
            self.dataset_exp_param.append(params[:, 80:144])
            self.dataset_idx += list(np.arange(0, min_num_frame-self.frames, 1) + base)
            base += min_num_frame


        self.dataset_audio = torch.Tensor(np.concatenate(self.dataset_audio))
        self.dataset_exp_param = torch.Tensor(np.concatenate(self.dataset_exp_param))
        self.dataset_idx = torch.Tensor(np.array(self.dataset_idx)).to(torch.long)
        
        print(max(self.dataset_idx))
        assert self.dataset_audio.shape[0] == self.dataset_exp_param.shape[0] 


    def __len__(self):
        return len(self.dataset_idx)

    def __getitem__(self, idx):
        audio = self.dataset_audio[self.dataset_idx[idx]:self.dataset_idx[idx]+self.frames]
        return audio, self.dataset_exp_param[self.dataset_idx[idx]:self.dataset_idx[idx]+self.frames]
