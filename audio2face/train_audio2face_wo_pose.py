'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-19 11:32:34
Email: haimingzhang@link.cuhk.edu.cn
Description: Train the audio2face network with only consider facial expressions
'''

import  torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model import TfaceGAN, NLayerDiscriminator
from dataset102 import ExpressionDataset
import argparse
import os, glob
import os.path as osp
import numpy as np
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Train_setting')
parser.add_argument('--audiopath', type=str, default='/content/FACIAL/examples/audio_preprocessed/train1.pkl') # Audio DeepSpeech features
parser.add_argument('--npzpath', type=str, default='/content/FACIAL/video_preprocess/train1_posenew.npz') # GT Deep3DFace params
parser.add_argument('--eval_audiopath', type=str, default=None)
parser.add_argument('--eval_npzpath', type=str, default=None)
parser.add_argument('--pretainpath_gen', type=str, default=None)
parser.add_argument('--savepath', type=str, default = './checkpoint/train1')

opt = parser.parse_args()


if not os.path.exists(opt.savepath):
    os.mkdir(opt.savepath)


def get_training_list(opt):
    audio_paths, npz_paths = [], []
    
    if opt.audiopath.endswith(".txt"):
        lines = open(opt.audiopath).read().splitlines()

        for line in lines:
            audio_paths.append(line)
            deep3dface_file = osp.join(osp.dirname(line), "deep3dface.npz")
            npz_paths.append(deep3dface_file)
        return audio_paths, npz_paths

    audio_paths.append(opt.audiopath)
    npz_paths.append(opt.npzpath)
    return audio_paths, npz_paths


audio_paths, npz_paths = get_training_list(opt)
print(len(audio_paths))


batchsz = 16
epochs = 1100

device = torch.device('cuda')
torch.manual_seed(123)

training_set = ExpressionDataset(audio_paths, npz_paths)
train_loader = DataLoader(training_set,
                          batch_size=batchsz,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
print(f"Training data length: {len(training_set)} / {len(train_loader)}...")

if opt.eval_npzpath is not None:
    val_dataset = ExpressionDataset([opt.eval_audiopath], [opt.eval_npzpath])
    val_dataset_loader = DataLoader(val_dataset,
                                    batch_size=batchsz,
                                    shuffle=False,
                                    drop_last=True,
                                    pin_memory=True)
    print(f"Val data length: {len(val_dataset)} / {len(val_dataset_loader)}...")


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_dict2tensorboard(tb_writer, loss_dict, global_step, prefix="train"):
    for key, value in loss_dict.items():
        tb_writer.add_scalar(f"{prefix}/{key}", value, global_step)


def eval_model(model_G, val_dataset_loader, criteon1, criteon):
    model_G.eval()

    with torch.no_grad():
        total_eval_loss = 0

        loss_dict = {'loss_s': 0, 'lossg_e': 0, 'lossg_em': 0}

        for (x, y) in val_dataset_loader:
            x, y = x.to(device), y.to(device)

            motiony = y[:,1:,:] - y[:,:-1,:]
            motiony = nn.functional.pad(motiony, (0, 0, 0, 1)) # make it has the same dimension with y

            yf = model_G(x, y[:,:1,:])

            motionlogits = yf[:,1:,:] - yf[:,:-1,:]
            motionlogits = nn.functional.pad(motionlogits, (0, 0, 0, 1)) # make it has the same dimension with y

            ## Initial state loss
            loss_s = 10 * (criteon1(yf[:,:1,:], y[:,:1,:]))
            
            # Expression loss
            lossg_e = 20 * criteon(yf[:,:,:], y[:,:,:])
            lossg_em = 200 * criteon(motionlogits[:,:,:], motiony[:,:,:]) # Expression motion loss

            lossG = loss_s + lossg_e + lossg_em

            loss_dict['loss_s'] += loss_s.item()
            loss_dict['lossg_e'] += lossg_e.item()
            loss_dict['lossg_em'] += lossg_em.item()

            total_eval_loss += lossG.item()
        
        avg_eval_loss = total_eval_loss / len(val_dataset_loader)

        loss_dict['loss_s'] = loss_dict['loss_s'] / len(val_dataset_loader)
        loss_dict['lossg_e'] = loss_dict['lossg_e'] / len(val_dataset_loader)
        loss_dict['lossg_em'] = loss_dict['lossg_em'] / len(val_dataset_loader)
        loss_dict['lossG'] = avg_eval_loss
    
    return loss_dict
            

def main():
    modelgen = TfaceGAN(pred_ch=64).to(device) # Generator
    modeldis = NLayerDiscriminator().to(device) # Discriminator

    if opt.pretainpath_gen is not None:
        modelgen.load_state_dict(torch.load(opt.pretainpath_gen))

    optimG = optim.Adam(modelgen.parameters(), lr=1e-4)
    optimD = optim.Adam(modeldis.parameters(), lr=1e-4)

    criteon1 = nn.L1Loss()
    criteon = nn.MSELoss()

    ## -------------------Start train-----------------
    # Create the tensorboard logger
    tb_writer = SummaryWriter(osp.join(opt.savepath, "logdir"))

    global_step = -1
    
    for epoch in range(0, epochs):
        for step, (x, y) in enumerate(train_loader):
            # x: Bx128x16x29 y: Bx128x64
            global_step += 1

            modelgen.train()
            x, y = x.to(device), y.to(device)
            motiony = y[:,1:,:] - y[:,:-1,:]
            # motiony = nn.functional.pad(motiony, (0, 0, 0, 1)) # make it has the same dimension with y

            ## dis
            set_requires_grad(modeldis, True)
            
            predr = modeldis(torch.cat([y, motiony], 1))
            lossr = criteon(torch.ones_like(predr), predr)

            # predr = modeldis(torch.cat([y, motiony], 2))
            # lossr = disc_criterion(predr, torch.ones_like(predr))

            ## Generator forward
            yf = modelgen(x, y[:,:1,:])

            motionlogits = yf[:,1:,:] - yf[:,:-1,:]
            # motionlogits = nn.functional.pad(motionlogits, (0, 0, 0, 1)) # make it has the same dimension with y

            # predf = modeldis(torch.cat([yf, motionlogits], 2).detach())
            # lossf = disc_criterion(predf, torch.zeros_like(predf))

            predf = modeldis(torch.cat([yf, motionlogits], 1).detach())
            lossf = criteon(torch.zeros_like(predf), predf)

            lossD = lossr + lossf
            optimD.zero_grad()
            lossD.backward()
            optimD.step()

            # generator
            set_requires_grad(modeldis, False)

            loss_dict = {}

            ## Initial state loss
            loss_s = 10 * (criteon1(yf[:, :1, :], y[:, :1, :]))
            
            # Expression loss
            lossg_e = 200 * criteon(yf[:, :, :], y[:, :, :]) 
            lossg_em = 100 * criteon(motionlogits[:, :, :], motiony[:, :, :]) # Expression motion loss
            
            ## GAN loss
            predf2 = modeldis(torch.cat([yf, motionlogits], 1))
            lossg_gan = 0.08 * criteon(torch.ones_like(predf2), predf2)

            # predf2 = modeldis(torch.cat([yf, motionlogits], 2))
            # lossg_gan = 0.05 * disc_criterion(predf2, torch.ones_like(predf2))

            lossG = loss_s + lossg_e + lossg_em + lossg_gan

            loss_dict["loss_s"] = loss_s
            loss_dict["lossg_e"] = lossg_e
            loss_dict["lossg_em"] = lossg_em
            loss_dict["lossg_gan"] = lossg_gan
            loss_dict["lossG"] = lossG

            loss_dict['lossD'] = lossD

            save_dict2tensorboard(tb_writer, loss_dict, global_step, "train")

            optimG.zero_grad()
            lossG.backward()
            optimG.step()

            if global_step % 50 == 0:
                print(f"[Epoch]: {epoch} gloabl_step: {global_step} ",
                      f"lossG: {lossG}, loss_s: {loss_s}, lossg_e: {lossg_e}, lossg_em: {lossg_em}, lossg_gan: {lossg_gan} ",
                      f"lossD: {lossD}")

        if opt.eval_npzpath is not None:
            ## ----------Start eval--------------------
            print(f"=====================================Start eval================================================")
            eval_loss = eval_model(modelgen, val_dataset_loader, criteon1, criteon)
            save_dict2tensorboard(tb_writer, eval_loss, global_step, "val")

        if epoch % 5 == 0:
            torch.save(modelgen.state_dict(), opt.savepath+'/Gen-'+str(epoch)+ "-" + str(global_step) + '.mdl')
            torch.save(modeldis.state_dict(), opt.savepath+'/Dis-'+str(epoch)+ "-" + str(global_step) + '.mdl')


if __name__ == '__main__':
    main()

