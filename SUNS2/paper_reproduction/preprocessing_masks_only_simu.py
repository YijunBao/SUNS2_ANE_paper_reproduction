# %%
import sys
import os
import random
import time
import glob
import numpy as np
import math
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

if __name__ == '__main__':
    list_name_video = ['noise15']
    # list_radius = [8,10,8,6] # 
    list_rate_hz = [10] # 
    # list_decay_time = [0.4, 0.5, 0.4, 0.75]
    Dimens = [(316,253)]
    list_nframes = [2000]
    num_Exp = 10
    list_Exp_ID = ['sim_'+str(x) for x in range(0,num_Exp)]
    # list_thred_std = [5,5,5,3]

    ind_video = 0
    name_video = list_name_video[ind_video]
    # %% setting parameters
    rate_hz = list_rate_hz[ind_video] # frame rate of the video
    Dimens = Dimens[ind_video] # lateral dimensions of the video
    nn = list_nframes[ind_video] # number of frames used for preprocessing. 
        # Can be slightly larger than the number of frames of a video
    # num_total = 6000 # number of frames used for CNN training. 
        # Can be slightly smaller than the number of frames of a video
    Mag = 1 # spatial magnification compared to ABO videos.

    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    # list_Exp_ID = [name_video+x for x in ID_part]
    # folder of the raw videos
    dir_video = os.path.join('E:\\simulation_CNMFE', name_video)
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = os.path.join(dir_video, 'GT Masks\\FinalMasks_')
    dir_parent = os.path.join(dir_video, 'complete_TUnCaT') # folder to save all the processed data
    # dir_network_input = os.path.join(dir_parent, 'network_input') # folder of the SNR videos
    # dir_mask = os.path.join(dir_parent, 'temporal_masks({})'.format(thred_std)) # foldr to save the temporal masks

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    (rows, cols) = Dimens # size of the original video
    rowspad = math.ceil(rows/8)*8  # size of the network input and output
    colspad = math.ceil(cols/8)*8

    # pre-processing for training
    for Exp_ID in list_Exp_ID: #
    #     # %% Pre-process video
    #     video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
    #         useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, prealloc=prealloc) #
    #     # %% Determine active neurons in all frames using FISSA
    #     file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
    #     generate_masks(video_input, file_mask, list_thred_ratio, dir_parent, Exp_ID)
    #     del video_input

        list_thred_ratio = list(range(6,9))
        from suns.PreProcessing.generate_masks import generate_masks_from_traces
        file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        generate_masks_from_traces(file_mask, list_thred_ratio, dir_parent, Exp_ID)
