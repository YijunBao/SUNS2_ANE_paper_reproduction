import os
import sys
import numpy as np
from scipy.io import savemat, loadmat
import h5py

from tuncat.run_TUnCaT import run_TUnCaT


if __name__ == '__main__':
    # %% Set the folder for the input and output data
    # A list of the name of the videos
    dir_masks = sys.argv[1] # "SUNS_TUnCaT_SF25/4816[1]th4/output_masks" # 
    tol_str = sys.argv[2]
    bin_str = sys.argv[3]
    # name_video = sys.argv[2] # 'lowBG=5e+03,poisson=1' # 

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
    dir_traces = os.path.join(dir_masks, 'TUnCaT_time'+'_tol='+tol_str+'_bin='+bin_str)
    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 

    # %% Set parameters
    # A list of tested alpha.
    list_alpha = [1]
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
    nbin = int(bin_str) # 1
    # The method of temporal downsampling. can be 'downsample', 'sum', or 'mean'
    bin_option = 'downsample' # 'sum' # 'mean' # 
    # Whether a flexible alpha strategy is used 
    # when the smallest alpha in "list_alpha" already caused over-regularization.
    flexible_alpha = True
    tol = float(tol_str)
    # Tolerance of the stopping condition in NMF.
    max_iter = 20000
    # Maximum number of iterations before timing out in NMF.

    # %% Run TUnCaT on the demo video
    for Exp_ID in list_Exp_ID:
        print(Exp_ID)
        # The file path (including file name) of the video.
        filename_video = os.path.join(dir_video, Exp_ID + '.h5')
        # The file path (including file name) of the neuron masks. 
        filename_masks = os.path.join(dir_masks, 'Output_Masks_' + Exp_ID + '.mat')
        filename_masks_resave = os.path.join(dir_traces, 'Output_Masks_' + Exp_ID + '.mat')
        
        try:
            file_masks = loadmat(filename_masks)
            Masks = file_masks['Masks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            Masks = np.array(file_masks['Masks']).astype('bool')
            file_masks.close()
        savemat(filename_masks_resave, {"FinalMasks": Masks}, do_compression=True)

        # run TUnCaT to calculate the unmixed traces of the marked neurons in the video
        traces_nmfdemix, list_mixout, traces, bgtraces, TUnCaT_time = \
            run_TUnCaT(Exp_ID, filename_video, filename_masks_resave, dir_traces, list_alpha, Qclip=Qclip, \
                th_pertmin=th_pertmin, epsilon=epsilon, th_residual=th_residual, nbin=nbin, bin_option=bin_option, \
                multi_alpha=multi_alpha, flexible_alpha=flexible_alpha, tol=tol, max_iter=max_iter)

        savemat(filename_masks_resave, {"Masks": Masks, "traces":traces_nmfdemix, "TUnCaT_time":TUnCaT_time}, do_compression=True)
