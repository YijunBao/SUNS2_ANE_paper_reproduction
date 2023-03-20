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

# %%
if __name__ == '__main__':
    list_name_video = ['blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps']
    # list_radius = [8,10,8,6] # 
    list_rate_hz = [10,15,30,5] # 
    # list_decay_time = [0.4, 0.5, 0.4, 0.75]
    Dimens = [(224,224),(80,80), (248,248),(120,88)]
    list_nframes = [90000, 9000, 116043, 3000]
    ID_part = ['_part11', '_part12', '_part21', '_part22']
    # list_Mag = [x/8 for x in list_radius]
    # list_thred_std = [5,5,5,3]

    ind_video = 1
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
    list_Exp_ID = [name_video+x for x in ID_part]
    # folder of the raw videos
    dir_video = os.path.join('E:\\data_CNMFE', name_video)
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = os.path.join(dir_video, 'GT Masks\\FinalMasks_')
    dir_parent = os.path.join(dir_video, 'complete_FISSA') # folder to save all the processed data
    # dir_network_input = os.path.join(dir_parent, 'network_input') # folder of the SNR videos
    # dir_mask = os.path.join(dir_parent, 'temporal_masks({})'.format(thred_std)) # foldr to save the temporal masks

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    (rows, cols) = Dimens # size of the original video
    rowspad = math.ceil(rows/8)*8  # size of the network input and output
    colspad = math.ceil(cols/8)*8

    # %% set pre-processing parameters
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    list_thred_ratio = list(range(3,10)) # [thred_std] # A list of SNR threshold used to determine when neurons are active.
    filename_TF_template = os.path.join(dir_video, name_video+'_spike_tempolate.h5')

    h5f = h5py.File(filename_TF_template,'r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}
    num_total = nn - Poisson_filt.size + 1

    # pre-processing for training
    for Exp_ID in list_Exp_ID: #
        # # %% Pre-process video
        # video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
        #     useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, prealloc=prealloc) #
        # # %% Determine active neurons in all frames using FISSA
        # file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        # generate_masks(video_input, file_mask, list_thred_ratio, dir_parent, Exp_ID)
        # del video_input

        list_thred_ratio = [2] # list(range(3,9))
        from suns.PreProcessing.generate_masks import generate_masks_from_traces
        file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        generate_masks_from_traces(file_mask, list_thred_ratio, dir_parent, Exp_ID)
