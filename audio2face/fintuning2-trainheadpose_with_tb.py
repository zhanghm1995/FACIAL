"""Finetune the audio2face network with tensorboard inside
"""
import  torch
from torch import optim, nn, autograd
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from model import TfaceGAN, NLayerDiscriminator
from dataset102 import Facial_Dataset
import argparse
import os, glob
import os.path as osp
import random, csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Train_setting')
parser.add_argument('--audiopath', type=str, default='/content/FACIAL/examples/audio_preprocessed/train1.pkl') # Audio DeepSpeech features
parser.add_argument('--npzpath', type=str, default='/content/FACIAL/video_preprocess/train1_posenew.npz') # GT Deep3DFace params
parser.add_argument('--eval_audiopath', type=str, default=None)
parser.add_argument('--eval_npzpath', type=str, default=None)
parser.add_argument('--cvspath', type=str, default = '/content/FACIAL/video_preprocess/train1_openface/train1_512_audio.csv')
parser.add_argument('--pretainpath_gen', type=str, default = '/content/FACIAL/audio2face/checkpoint/obama/Gen-20-0.0006273046686902202.mdl')
parser.add_argument('--savepath', type=str, default = './checkpoint/train1')
opt = parser.parse_args()


if not os.path.exists(opt.savepath):
    os.mkdir(opt.savepath)

audio_paths = []
audio_paths.append(opt.audiopath)
npz_paths = []
npz_paths.append(opt.npzpath)
cvs_paths = []
cvs_paths.append(opt.cvspath)

batchsz = 16
epochs = 1100

device = torch.device('cuda')
torch.manual_seed(1234)

training_set = Facial_Dataset(audio_paths, npz_paths)
train_loader = DataLoader(training_set,
                          batch_size=batchsz,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
print(f"Training data length: {len(training_set)} / {len(train_loader)}...")

if opt.eval_npzpath is not None:
    val_dataset = Facial_Dataset([opt.eval_audiopath], [opt.eval_npzpath], cvs_paths=None)
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

        loss_dict = {'loss_s': 0, 'lossg_e': 0, 'lossg_em': 0,
                     'loss_pose': 0, 'loss_posem': 0}

        for (x, y) in val_dataset_loader:
            x, y = x.to(device), y.to(device)

            motiony = y[:,1:,:] - y[:,:-1,:]

            yf = model_G(x, y[:,:1,:])

            yf[:, :, :7] = y[:, :, :7]

            motionlogits = yf[:,1:,:] - yf[:,:-1,:]

            ## Initial state loss
            loss_s = 10 * (criteon1(yf[:,:1,:6], y[:,:1,:6]) + 
                           criteon1(yf[:,:1,6], y[:,:1,6]) + 
                           criteon1(yf[:,:1,7:], y[:,:1,7:]))
            
            lossg_e = 20 * criteon(yf[:,:,7:], y[:,:,7:]) # Expression loss
            lossg_em = 200 * criteon(motionlogits[:,:,7:], motiony[:,:,7:]) # Expression motion loss

            loss_pose = 1 * criteon(yf[:,:,:6], y[:,:,:6])
            loss_posem = 10 * criteon(motionlogits[:,:,:6], motiony[:,:,:6])

            lossG = loss_s + lossg_e + lossg_em + loss_pose + loss_posem

            loss_dict['loss_s'] += loss_s.item()
            loss_dict['lossg_e'] += lossg_e.item()
            loss_dict['lossg_em'] += lossg_em.item()
            loss_dict['loss_pose'] += loss_pose.item()
            loss_dict['loss_posem'] += loss_posem.item()

            total_eval_loss += lossG.item()
        
        avg_eval_loss = total_eval_loss / len(val_dataset_loader)

        loss_dict['loss_s'] = loss_dict['loss_s'] / len(val_dataset_loader)
        loss_dict['lossg_e'] = loss_dict['lossg_e'] / len(val_dataset_loader)
        loss_dict['lossg_em'] = loss_dict['lossg_em'] / len(val_dataset_loader)
        loss_dict['loss_pose'] = loss_dict['loss_pose'] / len(val_dataset_loader)
        loss_dict['loss_posem'] = loss_dict['loss_posem'] / len(val_dataset_loader)
        loss_dict['lossG'] = avg_eval_loss
    
    return loss_dict
            

def main():
    lr = 1e-4
    modelgen = TfaceGAN().to(device) # Generator
    modeldis = NLayerDiscriminator().to(device) # Discriminator

    modelgen.load_state_dict(torch.load(opt.pretainpath_gen))

    optimG = optim.Adam(modelgen.parameters(), lr=lr*0.1)
    optimD = optim.Adam(modeldis.parameters(), lr=lr*0.1)

    criteon1 = nn.L1Loss()
    criteon = nn.MSELoss()

    ## -------------------Start train-----------------
    # Create the tensorboard logger
    tb_writer = SummaryWriter(osp.join(opt.savepath, "logdir"))

    global_step = -1
    
    for epoch in range(0, epochs):

        for step, (x, y) in enumerate(train_loader):
            # x: Bx128x16x29 y: Bx128x71
            global_step += 1

            modelgen.train()
            x, y = x.to(device), y.to(device)
            motiony = y[:,1:,:] - y[:,:-1,:]

            ## dis
            set_requires_grad(modeldis, True)

            predr = modeldis(torch.cat([y, motiony], 1))
            lossr = criteon(torch.ones_like(predr), predr)

            ## Generator forward
            yf = modelgen(x, y[:,:1,:])

            ## ----Donot consider head pose and eye blink
            yf[:, :, :7] = y[:, :, :7]

            motionlogits = yf[:,1:,:] - yf[:,:-1,:]
            
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
            loss_s = 10 * (criteon1(yf[:, :1, :6], y[:, :1, :6]) + 
                           criteon1(yf[:, :1, 6], y[:, :1, 6]) + 
                           criteon1(yf[:, :1, 7:], y[:, :1, 7:]))
            lossg_e = 20 * criteon(yf[:,:,7:], y[:,:,7:]) # Expression loss
            lossg_em = 200 * criteon(motionlogits[:,:,7:], motiony[:,:,7:]) # Expression motion loss
            
            loss_au = 0.5 * criteon(yf[:,:,6], y[:,:,6])
            loss_aum = 1 * criteon(motionlogits[:,:,6], motiony[:,:,6])

            loss_pose = 1 * criteon(yf[:,:,:6], y[:,:,:6])
            loss_posem = 10 * criteon(motionlogits[:,:,:6], motiony[:,:,:6])
            
            predf2 = modeldis(torch.cat([yf, motionlogits], 1))
            lossg_gan = criteon(torch.ones_like(predf2), predf2)

            lossG = loss_s + lossg_e + lossg_em + loss_au + loss_aum + loss_pose + loss_posem + 0.1*lossg_gan

            loss_dict["loss_s"] = loss_s
            loss_dict["lossg_e"] = lossg_e
            loss_dict["lossg_em"] = lossg_em
            loss_dict["loss_au"] = loss_au
            loss_dict["loss_aum"] = loss_aum
            loss_dict["loss_pose"] = loss_pose
            loss_dict["loss_posem"] = loss_posem
            loss_dict["lossg_gan"] = 0.1*lossg_gan
            loss_dict["lossG"] = lossG

            save_dict2tensorboard(tb_writer, loss_dict, global_step, "train")

            optimG.zero_grad()
            lossG.backward()
            optimG.step()

            if step % 60 == 0:
                print('epoch: ',epoch, 'global_step', global_step, ' loss_s: ',loss_s.item(),' lossg_e: ',lossg_e.item(), ' lossg_em: ',lossg_em.item())
                print(' loss_au: ',loss_au.item(),' loss_aum: ',loss_aum.item()) 
                print(' loss_pose: ',loss_pose.item(),' loss_posem: ',loss_posem.item()) 

        if opt.eval_npzpath is not None:
            ## ----------Start eval--------------------
            print(f"=====================================Start eval================================================")
            eval_loss = eval_model(modelgen, val_dataset_loader, criteon1, criteon)
            save_dict2tensorboard(tb_writer, eval_loss, global_step, "val")

        if epoch % 5 == 0:
            torch.save(modelgen.state_dict(), opt.savepath+'/Gen-'+str(epoch)+ "-" + str(global_step) + '.mdl')
            torch.save(modeldis.state_dict(), opt.savepath+'/Dis-'+str(epoch)+ "-" + str(global_step) +'.mdl')


if __name__ == '__main__':
    main()

