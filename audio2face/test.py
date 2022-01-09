"""Generate the facial parameters given audio features as input
"""
import os
import numpy as np
import torch
import argparse
import scipy
import copy
from scipy.io import wavfile
import pickle
from    model import TfaceGAN
import glob
from os.path import join, exists, abspath, dirname

parser = argparse.ArgumentParser(description='Test_setting')
parser.add_argument('--audiopath', type=str, default='../examples/audio_preprocessed/obama2.pkl')
parser.add_argument('--checkpath', type=str, default='./checkpoint/obama/Gen-20-0.0006273046686902202.mdl')
parser.add_argument('--outpath', type=str, default = '../examples/test-result')
parser.add_argument('--use_first_gt_params', type=bool, default=False)
parser.add_argument('--reference_gt_params', type=str, default='../gangqiang_video_preprocess/gangqiang_posenew.npz')

opt = parser.parse_args()

print("==================================")
for arg in vars(opt):
    print(arg.rjust(15) + " : " + str(getattr(opt, arg)))
print("==================================\n")


num_params = 71
out_path = opt.outpath

if not os.path.exists(out_path):
	os.makedirs(out_path)

audio_list = glob.glob(opt.audiopath)

for audio_path in audio_list:
    print(f"Start processing {audio_path}...")
    
    processed_audio = pickle.load(open(audio_path, 'rb'), encoding=' iso-8859-1')

    ## Load model
    modelgen = TfaceGAN().cuda()

    modelgen.load_state_dict(torch.load(opt.checkpath))
    modelgen.eval()

    processed_audio = torch.Tensor(processed_audio)
    audioname = audio_path.split('/')[-1].replace('.pkl', '')

    faceparams = np.zeros((processed_audio.shape[0], num_params), float) # Nx71

    frames_out_path = os.path.join(out_path, audioname+'.npz')

    firstpose = torch.zeros([1, num_params], dtype=torch.float32).unsqueeze(0)
    if opt.use_first_gt_params:
        reference_params = np.load(open(opt.reference_gt_params, 'rb'))['face']
        std1 = np.std(reference_params, axis=0)
        mean1 = np.mean(reference_params,axis=0)
        for i in range(6):
            reference_params[:,i] = (reference_params[:,i] - mean1[i]) / std1[i]
        
        aublink = np.zeros((reference_params.shape[0], 1))

        first_pose = np.concatenate((reference_params[:,:6], aublink, reference_params[:,7:71]), axis=1)
        first_pose = first_pose.astype(np.float32)
        firstpose = torch.from_numpy(first_pose[:1, ...]).unsqueeze(0)
        

    with torch.no_grad():
        for i in range(0, processed_audio.shape[0]-127, 127):
            audio = processed_audio[i:i+128, :, :].unsqueeze(0).cuda()

            _faceparam = modelgen(audio, firstpose.cuda())

            firstpose = _faceparam[:, 127:128, :]
            faceparams[i:i+128, :] = _faceparam[0,:,:].cpu().numpy()

            # last audio sequence
            if i+127 >= processed_audio.shape[0]-127:
                j = processed_audio.shape[0]-128
                audio = processed_audio[j:j+128,:,:].unsqueeze(0).cuda()
                firstpose = _faceparam[:,j-i:j-i+1,:]
                _faceparam = modelgen(audio, firstpose.cuda())
                faceparams[j:j+128,:] = _faceparam[0,:,:].cpu().numpy()

        np.savez(frames_out_path, face=faceparams)


