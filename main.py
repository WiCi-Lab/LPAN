# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 09:55:32 2022
Modified on Dec. 2 2023

@author: Victor
"""


from __future__ import print_function
import argparse
from math import log10
from os.path import exists, join, basename
from os import makedirs, remove

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import math
import matplotlib.pyplot as plt

# LPAN
from LPAN import LPAN
# LPAN_L
from LPAN_L import LPAN_L

torch.backends.cudnn.benchmark = True

# Training settings 
parser = argparse.ArgumentParser(description='LPAN/LPAN-L')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', default=True,action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
    
# Loading data
print('===> Loading datasets')
import h5py
path="E:/LS_64_256R_6users_32pilot.mat"
with h5py.File(path, 'r') as file:
    train_h = np.transpose(np.array(file['output_da']))
    train_h = train_h.transpose([0,3,1,2])
    test_h = np.transpose(np.array(file['output_da_test']))
    test_h = test_h.transpose([0,3,1,2])

    train_y = np.transpose(np.array(file['input_da']))
    train_y = train_y.transpose([0,3,1,2])
    test_y = np.transpose(np.array(file['input_da_test']))
    test_y = test_y.transpose([0,3,1,2])
    
train_y=torch.Tensor(train_y)
test_y=torch.Tensor(test_y)
train_h=torch.Tensor(train_h)
test_h=torch.Tensor(test_h)

train_set = Data.TensorDataset(train_y, train_h) 
val_set = Data.TensorDataset(test_y, test_h) 

# Release Memory
del train_h
del test_h
del train_y
del test_y

batchsize = 64

train_set = torch.utils.data.DataLoader(train_set, batch_size= batchsize, shuffle=True)
val_set = torch.utils.data.DataLoader(val_set, batch_size= batchsize, shuffle=True)

def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict.cpu().double()-target.cpu().double()), 2) + 1e-5)) # epsilon=1e-3

# Building model
print('===> Building model')
model = LPAN().to(device)
# if you want to train the LPAN-L model
# model = LPAN_L().to(device)
optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
print (model)
if cuda:
    model = model.cuda()

def checkpoint(epoch):
    if not exists(opt.checkpoint):
        makedirs(opt.checkpoint)
    model_out_path = "model/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

ne=[]
opt.nEpochs=100
test_ne=[]

#Learning rate decay 
def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    epochs = opt.nEpochs
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

scaler = torch.cuda.amp.GradScaler()

# Traning model
for epoch in range(1, opt.nEpochs + 1):
    lr = adjust_learning_rate(optimizer, epoch,1e-3,5e-6)    
    print(lr)
    epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    model.train()
    nm=[]
    for iteration, (pilot,batch) in enumerate(train_set, 1):
        
        a,b,c,d=batch.size()
        HR_2_target = np.zeros([a,b,c,int(d/4)],dtype=float)
        HR_4_target = np.zeros([a,b,c,int(d/2)],dtype=float)

        for i in range(1,d,4):
            HR_2_target[:,:,:,int(math.ceil(i/4)-1)]=batch[:,:,:,i].cpu().numpy()
            
        for i in range(1,d,2):
            HR_4_target[:,:,:,int(math.ceil(i/2)-1)]=batch[:,:,:,i].cpu().numpy()
            
        HR_8_target=batch
                   
        if cuda:
            LR = Variable(pilot).cuda()
            HR_2_target = Variable(torch.from_numpy(HR_2_target)).cuda()
            HR_4_target = Variable(torch.from_numpy(HR_4_target)).cuda()
            HR_8_target = Variable(HR_8_target).cuda()
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            HR_2, HR_4, HR_8 = model(LR)
            
            loss1 = CharbonnierLoss(HR_2.float(), HR_2_target.float())
            loss2 = CharbonnierLoss(HR_4.float(), HR_4_target.float())
            loss3 = CharbonnierLoss(HR_8.float(), HR_8_target.float())
            loss = (loss1+loss2+loss3).cuda()
        
        epoch_loss += loss.item()
        epoch_loss1 += loss1.item()
        epoch_loss2 += loss3.item()
        
        #Adaptive mixed precision method
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
           
        nm.append(loss3.item())
        
        
        if iteration%50==0:
            nmsei=np.zeros([a, 1])
            for i1 in range(a):
                nmsei[i1] = np.sum(np.square(np.abs(HR_8[i1,:,:,:].cpu().detach().numpy()-batch[i1,:,:,:].cpu().numpy()))) / np.sum(np.square(np.abs(batch[i1,:,:,:].cpu().numpy())))
                tr_nmse = np.sum(nmsei) / a 
                
            print("===> Epoch[{}]({}/{}): loss1: {:.4f} loss3: {:.4f} nmse: {:.4f}".format(
            epoch, iteration, len(train_set), loss1.item(),loss3.item(),10*np.log10(tr_nmse)))
    nmse_n=np.sum(nm)/len(nm)
    print("===> Avg1. Loss: {:.4f}, Avg2. Loss: {:.4f}".format(epoch_loss1 / len(train_set),epoch_loss2 / len(train_set)))
    ne.append(nmse_n)
    
    # Model validation
    nm1=[]
    nl1=[]
    model.eval()
    with torch.no_grad():
        for iteration, (pilot,batch) in enumerate(val_set, 1):
                
            a,b,c,d=batch.size()
                   
            if cuda:
                LR = Variable(pilot).cuda()

            HR_2, HR_4, HR_8 = model(LR)
            nmsei=np.zeros([a, 1])
            for i1 in range(a):
                nmsei[i1] = np.sum(np.square(np.abs(HR_8[i1,:,:,:].cpu().detach().numpy()-batch[i1,:,:,:].cpu().numpy()))) / np.sum(np.square(np.abs(batch[i1,:,:,:].cpu().numpy())))
            te_nmse = np.sum(nmsei) / a
            nm1.append(te_nmse)
            
            
        nmse=np.sum(nm1)/len(nm1)
        nmse_db=10*np.log10(nmse)
        print("===> test-NMSE: {:.8f} dB".format(nmse_db))
        test_ne.append(nmse_db)

# Release Memory
del val_set
del train_set


# Testing Model
model.eval()
path="E:/LS_64_256R_test_6users_32pilot.mat"
with h5py.File(path, 'r') as file:
    test_h = np.transpose(np.array(file['Hd']))
    test_h = test_h.transpose([0,3,1,2])
with h5py.File(path, 'r') as file:
    test_y = np.transpose(np.array(file['Yd']))
    test_y = test_y.transpose([0,3,1,2])
    
print('load done')

test_y=torch.Tensor(test_y)
test_h=torch.Tensor(test_h)

test_set = Data.TensorDataset(test_y, test_h) 

del test_y
del test_h

batchsize = 20

test_set = torch.utils.data.DataLoader(test_set, batch_size= batchsize, shuffle=False)
nmse1_snr=[]
test_nmse=[]
from time import time
for i,batch in enumerate(test_set, 1):
    input, target = batch[0].to(device), batch[1].to(device)
    
    a,b,c,d=target.size()
    
               
    if cuda:
        LR = Variable(input).cuda()
    
    prediction1,prediction2,prediction3 = model(input)
   
    nmse = np.zeros([prediction3.shape[0], 1])
    for i2 in range(prediction3.shape[0]):
        nmse[i2] = np.sum(np.square(np.abs(prediction3[i2, :, :, :].cpu().detach().numpy() - target[i2, :, :, :].cpu().detach().numpy()))) / np.sum(
          np.square(np.abs(target[i2, :, :, :].cpu().detach().numpy())))
    tenmse = np.mean(nmse)
    test_nmse.append(tenmse)
        
        
    
    if i%50==0:
        nmse1=np.mean(test_nmse)
        nmse1_snr.append(nmse1)
        test_nmse=[]
plt.figure()
nmse1_db=10*np.log10(nmse1_snr)
snrs = np.linspace(-10,30,9)
pilots = np.linspace(32,128,7)


plt.plot(snrs, nmse1_db)
plt.grid(True) 
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.show()



    
