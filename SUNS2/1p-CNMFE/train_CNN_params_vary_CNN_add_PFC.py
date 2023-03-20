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
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PreProcessing.preprocessing_functions import preprocess_video
from suns.PreProcessing.generate_masks import generate_masks
from suns.Network.train_CNN_params_vary_CNN import train_CNN, parameter_optimization_cross_validation

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config = config)

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
    n_depth = int(sys.argv[1])
    n_channel = int(sys.argv[2])
    skip = eval(sys.argv[3])
    activation = sys.argv[4]
    double = eval(sys.argv[5])
    sub_folder = sys.argv[6] # + '+BGlayer' # + '_2out' 
    # max_eid = int(sys.argv[7])
    thred_std = int(sys.argv[7]) # SNR threshold used to determine when neurons are active
    sub_added = sys.argv[8] # sub folder for added neurons

    num_train_per = 2400 # Number of frames per video used for training 
    BATCH_SIZE = 20 # Batch size for training 
    NO_OF_EPOCHS = 200 # Number of epoches used for training 
    batch_size_eval = 200 # batch size in CNN inference

    useSF=True # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    med_subtract=False # True if the spatial median of every frame is subtracted before temporal filtering.
        # Can only be used when spatial filtering is not used. 
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.
    useWT=False # True if using additional watershed
    load_exist=True # True if using temp files already saved in the folders
    use_validation = True # True to use a validation set outside the training set
    # Cross-validation strategy. Can be "leave_one_out" or "train_1_test_rest"
    cross_validation = "leave_one_out"
    # Params_loss = {'DL':1, 'BCE':0, 'FL':100, 'gamma':1, 'alpha':0.25} # Parameters of the loss function
    Params_loss = {'DL':1, 'BCE':20, 'FL':0, 'gamma':1, 'alpha':0.25} # Parameters of the loss function

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
    list_Exp_ID = [name_video+x+'_added' for x in ID_part]
    # folder of the raw videos
    dir_video = os.path.join('E:\\data_CNMFE', name_video+'_original_masks', sub_added)
    # dir_video = os.path.join('E:\\data_CNMFE', name_video+'_added_blockwise_weighted_sum_unmask', sub_added)
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = os.path.join(dir_video, 'GT Masks\\FinalMasks_')
    dir_parent = os.path.join(dir_video, 'complete_FISSA') # folder to save all the processed data
    dir_network_input = os.path.join(dir_parent, 'network_input') # folder of the SNR videos
    dir_mask = os.path.join(dir_parent, 'temporal_masks({})'.format(thred_std)) # foldr to save the temporal masks
    dir_sub = sub_folder
    weights_path = os.path.join(dir_parent, dir_sub, 'Weights') # folder to save the trained CNN
    training_output_path = os.path.join(dir_parent, dir_sub, 'training output') # folder to save the loss functions during training
    dir_output = os.path.join(dir_parent, dir_sub, 'output_masks') # folder to save the optimized hyper-parameters
    dir_temp = os.path.join(dir_parent, dir_sub, 'temp') # temporary folder to save the F1 with various hyper-parameters

    if not os.path.exists(dir_network_input):
        os.makedirs(dir_network_input) 
    if not os.path.exists(weights_path):
        os.makedirs(weights_path) 
    if not os.path.exists(training_output_path):
        os.makedirs(training_output_path) 
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp) 

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    (rows, cols) = Dimens # size of the original video
    rowspad = math.ceil(rows/8)*8  # size of the network input and output
    colspad = math.ceil(cols/8)*8

    # %% set pre-processing parameters
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    list_thred_ratio = list(range(2,6)) # [thred_std] # A list of SNR threshold used to determine when neurons are active.
    filename_TF_template = os.path.join(dir_video, name_video+'_spike_tempolate.h5')

    h5f = h5py.File(filename_TF_template,'r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}
    num_total = nn - Poisson_filt.size + 1

    # %% set the range of post-processing hyper-parameters to be optimized in
    # minimum area of a neuron (unit: pixels in ABO videos). must be in ascend order
    # list_minArea = list(range(20,75,10)) 
    list_minArea = list(range(40,105,10)) 
    # average area of a typical neuron (unit: pixels in ABO videos)
    list_avgArea = [100] # [72] # list(range(30,58,5)) # 
    # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
    list_thresh_pmap = list(range(40,256,20))
    # threshold to binarize the neuron masks. For each mask, 
    # values higher than "thresh_mask" times the maximum value of the mask are set to one.
    thresh_mask = 0.5
    # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels in ABO videos)
    thresh_COM0 = 1
    # maximum COM distance of two masks to be considered the same neuron (unit: pixels in ABO videos)
    list_thresh_COM = list(np.arange(2, 6, 0.5)) 
    # list_thresh_COM = list(np.arange(1, 5, 0.5)) 
    # minimum IoU of two masks to be considered the same neuron
    list_thresh_IOU = [0.5] 
    # minimum consecutive number of frames of active neurons
    list_cons = list(range(1, 8, 1)) 

    # adjust the units of the hyper-parameters to pixels in the test videos according to relative magnification
    list_minArea= list(np.round(np.array(list_minArea) * Mag**2))
    list_avgArea= list(np.round(np.array(list_avgArea) * Mag**2))
    thresh_COM0= thresh_COM0 * Mag
    list_thresh_COM= list(np.array(list_thresh_COM) * Mag)
    # adjust the minimum consecutive number of frames according to different frames rates between ABO videos and the test videos
    # list_cons=list(np.round(np.array(list_cons) * rate_hz/30).astype('int'))

    # dictionary of all fixed and searched post-processing parameters.
    Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
            'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
            'thresh_mask': thresh_mask, 'list_cons': list_cons}
    print(Params_set)


    # pre-processing for training
    for Exp_ID in list_Exp_ID: #
        # %% Pre-process video
        video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, prealloc=prealloc) #
        # %% Determine active neurons in all frames using FISSA
        file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        generate_masks(video_input, file_mask, list_thred_ratio, dir_parent, Exp_ID)
        del video_input

        # list_thred_ratio = list(range(10,22,2))
        # from suns.PreProcessing.generate_masks import generate_masks_from_traces
        # file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        # generate_masks_from_traces(file_mask, list_thred_ratio, dir_parent, Exp_ID)

    # %% CNN training
    for CV in range(0,nvideo): # [3]: # 
        if cross_validation == "leave_one_out":
            list_Exp_ID_train = list_Exp_ID.copy()
            list_Exp_ID_val = [list_Exp_ID_train.pop(CV)]
        else: # cross_validation == "train_1_test_rest"
            list_Exp_ID_val = list_Exp_ID.copy()
            list_Exp_ID_train = [list_Exp_ID_val.pop(CV)]
        if not use_validation:
            list_Exp_ID_val = None # Afternatively, we can get rid of validation steps
        file_CNN = os.path.join(weights_path,'Model_CV{}.h5'.format(CV))
        results = train_CNN(dir_network_input, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
            BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, (rowspad, colspad), Params_loss,\
            n_depth=n_depth, n_channel=n_channel, skip=skip, activation=activation, double=double)

        # save training and validation loss after each eopch
        f = h5py.File(os.path.join(training_output_path,"training_output_CV{}.h5".format(CV)), "w")
        f.create_dataset("loss", data=results.history['loss'])
        f.create_dataset("dice_loss", data=results.history['dice_loss'])
        if use_validation:
            f.create_dataset("val_loss", data=results.history['val_loss'])
            f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
        f.close()

    # %% parameter optimization
    parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, \
        (rows, cols), (rowspad, colspad), dir_network_input, weights_path, dir_GTMasks, dir_temp, dir_output, \
        batch_size_eval, useWT=useWT, useMP=True, load_exist=load_exist, \
        n_depth=n_depth, n_channel=n_channel, skip=skip, activation=activation, double=double) #  max_eid=max_eid,
