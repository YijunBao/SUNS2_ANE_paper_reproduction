import os
import sys
import numpy as np
from scipy.io import savemat, loadmat
import h5py
import cv2

from tuncat.run_TUnCaT import run_TUnCaT


if __name__ == '__main__':
    # %% Set the folder for the input and output data
    # A list of the name of the videos
    # dir_masks = "SUNS_TUnCaT_SF25/4816[1]th4/output_masks" # sys.argv[1] # 
    dir_masks = 'DeepWonder_scale_full'

    list_Exp_ID = ['c25_59_228','c27_12_326','c28_83_210',
                'c25_163_267','c27_114_176','c28_161_149',
                'c25_123_348','c27_122_121','c28_163_244']
    # The folder containing the videos
    dir_parent = 'E:\\1photon-small\\added_refined_masks\\' 
    # The folder name (excluding the file name) containing the video
    dir_video = dir_parent
    # The folder name (excluding the file name) containing the neuron masks
    dir_masks = os.path.join(dir_parent,dir_masks)
    # The folder to save the unmixed traces.
    dir_traces = os.path.join(dir_masks, 'TUnCaT')
    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 

    # %% Set parameters
    # A list of tested alpha.
    list_alpha = [] # [1] # 
    # If there are multiple elements in "list_alpha", whether consider them as independent trials.
    multi_alpha = True
    # False means the largest element providing non-trivial output traces will be used, 
    # which can be differnt for different neurons. It must be sorted in ascending order.
    # True means each element will be tested and saved independently.
    # Traces lower than this quantile are clipped to this quantile value.
    Qclip = 0
    # The minimum value of the input traces after scaling and shifting. 
    epsilon = 0
    # Maximum pertentage of unmixed traces equaling to the trace minimum.
    th_pertmin = 1
    # If th_residual > 0, The redisual of unmixing should be smaller than this value.
    th_residual = 0
    # The temporal downsampling ratio.
    nbin = 1
    # The method of temporal downsampling. can be 'downsample', 'sum', or 'mean'
    bin_option = 'downsample' # 'sum' # 'mean' # 
    # Whether a flexible alpha strategy is used 
    # when the smallest alpha in "list_alpha" already caused over-regularization.
    flexible_alpha = True

    # %% Run TUnCaT on the demo video
    for Exp_ID in list_Exp_ID:
        print(Exp_ID)
        # The file path (including file name) of the video.
        filename_video = os.path.join(dir_video, Exp_ID + '.h5')
        # The file path (including file name) of the neuron masks. 
        filename_masks = os.path.join(dir_masks, 'seg_30_' + Exp_ID + '_post.mat')
        filename_masks_resave = os.path.join(dir_traces, 'Output_Masks_' + Exp_ID + '.mat')
        
        try:
            file_masks = loadmat(filename_masks)
            Masks = file_masks['network_A_filt'].astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            Masks = np.array(file_masks['network_A_filt']).transpose([2,1,0]).astype('bool')
            file_masks.close()

        # Resize the masks
        Lxs = Lys = 50
        N = Masks.shape[2]
        Masks_resize = np.zeros((Lys, Lxs, N), 'bool')
        for n in range(N):
            Masks_resize[:,:,n] = cv2.resize(Masks[n].astype('uint8'), (Lxs, Lys))
        savemat(filename_masks_resave, {"FinalMasks": Masks_resize}, do_compression=True)

        # run TUnCaT to calculate the unmixed traces of the marked neurons in the video
        traces_nmfdemix, list_mixout, traces, bgtraces, time_TUnCaT = \
            run_TUnCaT(Exp_ID, filename_video, filename_masks_resave, dir_traces, list_alpha, Qclip, \
            th_pertmin, epsilon, th_residual, nbin, bin_option, multi_alpha, flexible_alpha)

        savemat(filename_masks_resave, {"Masks": Masks, "traces":traces_nmfdemix}, do_compression=True)
