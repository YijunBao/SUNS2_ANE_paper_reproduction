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
from suns.PreProcessing.generate_masks import generate_masks

# %%
if __name__ == '__main__':
    list_name_video = ['blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps']
    # list_radius = [8,10,8,6] # 
    list_rate_hz = [10,15,7.5,5] # 
    # list_decay_time = [0.4, 0.5, 0.4, 0.75]
    Dimens = [(120,120),(80,80), (88,88),(192,240)]
    list_nframes = [6000, 9000, 9000, 1500]
    ID_part = ['_part11', '_part12', '_part21', '_part22']
    # list_Mag = [x/8 for x in list_radius]
    # list_thred_std = [5,5,5,3]

    # sys.argv = ['py', '3', '4', '[1]', 'elu', 'True', '4816[1]']
    # n_depth = int(sys.argv[1])
    # n_channel = int(sys.argv[2])
    # skip = eval(sys.argv[3])
    # activation = sys.argv[4]
    # double = eval(sys.argv[5])
    # sub_folder = sys.argv[6] # + '+BGlayer' # + '_2out' 
    # sub_folder = '4816[1]th4'
    # max_eid = int(sys.argv[7])
    ind_video = int(sys.argv[1]) #  0
    folder_method = sys.argv[2] # 'complete_TUnCaT'
    name_video = list_name_video[ind_video]

    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = [name_video+x for x in ID_part]
    # folder of the raw videos
    dir_video = os.path.join('E:\\data_CNMFE', name_video)
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    # dir_GTMasks = os.path.join(dir_video, 'GT Masks\\FinalMasks_')
    # dir_parent = os.path.join(dir_video, folder_method) # folder to save all the processed data
    dir_network_input = os.path.join(dir_video, 'complete_TUnCaT\\network_input') # folder of the SNR videos
    # dir_sub = sub_folder
    dir_output_masks = os.path.join(dir_video, folder_method, 'output_masks') # folder to save the optimized hyper-parameters
    # dir_output_masks = os.path.join(dir_video, 'GT Masks') # folder to save the optimized hyper-parameters

    # pre-processing for training
    for Exp_ID in list_Exp_ID: #
        # %% Pre-process video
        # video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
        #     useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, prealloc=prealloc) #
        h5_video = os.path.join(dir_network_input, Exp_ID + '.h5')
        h5_file = h5py.File(h5_video,'r')
        video_input = h5_file['network_input']
        # %% Determine active neurons in all frames using FISSA
        # file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        file_mask = os.path.join(dir_output_masks, 'FinalMasks_' + Exp_ID + '.mat')
        generate_masks(video_input, file_mask, [], dir_output_masks, Exp_ID)
        del video_input
        h5_file.close()

        # list_thred_ratio = list(range(10,22,2))
        # from suns.PreProcessing.generate_masks import generate_masks_from_traces
        # file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        # generate_masks_from_traces(file_mask, list_thred_ratio, dir_parent, Exp_ID)
