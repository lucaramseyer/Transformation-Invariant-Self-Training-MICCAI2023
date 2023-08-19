#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:01:40 2022

@author: negin
"""
from torchvision import transforms
from nets import scSE_Net_VGG16 as Net

Net1 = Net

Categories = ['Spectralis_0','Spectralis_1','Spectralis_2','Spectralis_3']
Learning_Rates_init = [0.001]
epochs = 100
batch_size = 2
size = 'Determined in dataset method'

Dataset_Path_Train = '/storage/homefs/lr13y079/Master_Thesis/Transformation-Invariant-Self-Training-MICCAI2023/TrainIDs_RETOUCH_DA2/'
Dataset_Path_SemiTrain = ''
Dataset_Path_Test = ''
mask_folder = ''
Results_path = '../DA_ENCORE_RETOUCH_MICCAI23_Results/'
Visualization_path = 'visualization_RETOUCH/'
Checkpoint_path = 'checkpoints_RETOUCH/'
CSV_path = 'CSVs_RETOUCH/'
project_name = "ENCORE_RETOUCH_DA_MIC23_ST4"

load = False
load_path = ''


net_name = 'scSE_Net__'
strategy = 'Supervised'
test_per_epoch = 2

# Blanks:
hard_label_thr = ''
ensemble_batch_size = ''
SemiSup_initial_epoch = ''
image_transforms = ''
affine = ''
affine_transforms = ''
LW = ''
GCC = ''

EMA_decay = ''
Alpha = ''
supervised_share = ''



