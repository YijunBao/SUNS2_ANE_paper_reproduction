# %%
import sys
import os
# import random
# import time
# import glob
import numpy as np
import math
import h5py
# from scipy.io import savemat, loadmat
# import multiprocessing as mp

sys.path.insert(1, '..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PreProcessing.preprocessing_functions import preprocess_video
from suns.train_CNN_params import train_CNN, parameter_optimization_cross_validation
# from suns.Network.train_CNN_params_vary_CNN import train_CNN, parameter_optimization_cross_validation

sys.argv = ['py', '5', '25', '3', 'TUnCaT']
unmix = sys.argv[4] # 'TUnCaT' # 'FISSA' 
if unmix.upper() == 'FISSA':
    from suns.PreProcessing.generate_masks_fissa import generate_masks
else:
    from suns.PreProcessing.generate_masks_tuncat import generate_masks

import tensorflow as tf
tf_version = int(tf.__version__[0])
if tf_version == 1:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config = config)
else: # tf_version == 2:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# %%
if __name__ == '__main__':
    # n_depth = int(sys.argv[1])
    # n_channel = int(sys.argv[2])
    # skip = eval(sys.argv[3])
    # activation = sys.argv[4]
    # double = eval(sys.argv[5])
    thred_std = int(sys.argv[1]) # SNR threshold used to determine when neurons are active
    sub_folder = '4816[1]th' + sys.argv[1]
    gauss_filt_size = int(sys.argv[2]) # standard deviation of the spatial Gaussian filter in pixels
    max_eid = int(sys.argv[3])

    # %% setting parameters
    rate_hz = 20 # frame rate of the video
    Dimens = (480,480) # lateral dimensions of the video
    nn = 2100 # number of frames used for preprocessing. 
        # Can be slightly larger than the number of frames of a video
    # num_total = 6000 # number of frames used for CNN training. 
        # Can be slightly smaller than the number of frames of a video
    Mag = 1 # spatial magnification compared to ABO videos.

    num_train_per = 200 # Number of frames per video used for training 
    BATCH_SIZE = 20 # Batch size for training 
    NO_OF_EPOCHS = 200 # Number of epoches used for training 
    batch_size_eval = 200 # batch size in CNN inference

    useSF=gauss_filt_size>0 # True # True if spatial filtering is used in pre-processing.
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
    # cross_validation = "use_all"
    # Params_loss = {'DL':1, 'BCE':0, 'FL':100, 'gamma':1, 'alpha':0.25} # Parameters of the loss function
    Params_loss = {'DL':1, 'BCE':20, 'FL':0, 'gamma':1, 'alpha':0.25} # Parameters of the loss function

    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = [ 'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
                    'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']
    # folder of the raw videos
    # dir_video = '../../data/data_TENASPIS/original_masks' 
    dir_video = '../../data/data_TENASPIS/added_refined_masks' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = os.path.join(dir_video, 'GT Masks/FinalMasks_')
    if not useSF:
        unmix = unmix + '_noSF'
    dir_parent = os.path.join(dir_video, 'complete_'+unmix) # folder to save all the processed data
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
    gauss_filt_size = gauss_filt_size*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    list_thred_ratio = list(range(2,7)) # [thred_std] # A list of SNR threshold used to determine when neurons are active.
    filename_TF_template = os.path.join(dir_video, 'TENASPIS_spike_tempolate.h5')

    h5f = h5py.File(filename_TF_template,'r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}
    num_total = 1500 # nn - Poisson_filt.size + 1

    # %% set the range of post-processing hyper-parameters to be optimized in
    # minimum area of a neuron (unit: pixels in ABO videos). must be in ascend order
    list_minArea = list(range(80,250,20)) # list(range(100,340,20)) # 
    # average area of a typical neuron (unit: pixels in ABO videos)
    list_avgArea = [228] # [305] # 
    # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
    list_thresh_pmap = list(range(40,256,20))
    # threshold to binarize the neuron masks. For each mask, 
    # values higher than "thresh_mask" times the maximum value of the mask are set to one.
    thresh_mask = 0.5
    # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels in ABO videos)
    thresh_COM0 = 2
    # maximum COM distance of two masks to be considered the same neuron (unit: pixels in ABO videos)
    list_thresh_COM = list(np.arange(4, 10, 1))
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

        # %% Determine active neurons in all frames using TUnCaT
        file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        generate_masks(video_input, file_mask, list_thred_ratio, dir_parent, Exp_ID)
        del video_input

    # %% CNN training
    for CV in range(0,nvideo): # [0]: # 
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
            BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, (rowspad, colspad), Params_loss)
        # results = train_CNN(dir_network_input, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
        #     BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, (rowspad, colspad), Params_loss,\
        #     n_depth=n_depth, n_channel=n_channel, skip=skip, activation=activation, double=double)

        # save training and validation loss after each eopch
        f = h5py.File(os.path.join(training_output_path,"training_output_CV{}.h5".format(CV)), "w")
        f.create_dataset("loss", data=results.history['loss'])
        f.create_dataset("dice_loss", data=results.history['dice_loss'])
        if use_validation:
            f.create_dataset("val_loss", data=results.history['val_loss'])
            f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
        f.close()

    # %% parameter optimization
    # parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, \
    #     (rows, cols), dir_network_input, weights_path, dir_GTMasks, dir_temp, dir_output, \
    #     batch_size_eval, useWT=useWT, useMP=True, load_exist=load_exist, max_eid=max_eid) # 
    # # parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, \
    # #     (rows, cols), (rowspad, colspad), dir_network_input, weights_path, dir_GTMasks, dir_temp, dir_output, \
    # #     batch_size_eval, useWT=useWT, useMP=True, load_exist=load_exist, max_eid=max_eid, \
    # #     n_depth=n_depth, n_channel=n_channel, skip=skip, activation=activation, double=double) # 
