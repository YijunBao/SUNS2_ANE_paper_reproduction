import scipy.io as sio
import os
import numpy as np
import h5py
import sys
# from skimage import filters
from scipy.ndimage import gaussian_filter
import time
import multiprocessing as mp

sys.path.insert(1, '../SUNS2') # the path containing "suns" folder
from suns.PreProcessing.preprocessing_functions import preprocess_video
from frame_weight_blockwise_mm import frame_weight_blockwise
from find_missing_blockwise_mm import find_missing_blockwise
from traces_from_masks_mmap import traces_from_masks_mmap
# data_ind = 0; dir_masks = os.path.join('SUNS_TUnCaT_SF25','4816[1]th5','output_masks');
# data_ind = 1; dir_masks = os.path.join('SUNS_TUnCaT_SF25','4816[1]th4','output_masks');
# data_ind = 2; dir_masks = os.path.join('SUNS_TUnCaT_SF25','4816[1]th4','output_masks');
# data_ind = 3; dir_masks = os.path.join('SUNS_TUnCaT_SF50','4816[1]th4','output_masks')


if __name__ == "__main__":
    dir_masks = sys.argv[1] # "SUNS_TUnCaT_SF25/4816[1]th6/output_masks" # 

    list_Exp_ID = [ 'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
                    'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']
    num_Exp = len(list_Exp_ID)
    nframes = 2100

    avg_radius = 9 # added_refined_masks
    # avg_radius = 10 # original_masks
    r_bg_ratio = 3
    leng = r_bg_ratio * avg_radius
    th_IoU_split = 0.5
    thj_inclass = 0.4
    thj = 0.7
    useMP = True

    # dir_parent = '../data/data_TENASPIS/original_masks'
    dir_parent = '../data/data_TENASPIS/added_refined_masks'
    dir_video = dir_parent
    dir_masks = os.path.join(dir_parent,dir_masks)
    dir_add_new = os.path.join(dir_masks,'add_new_blockwise')
    if not os.path.exists(dir_add_new):
        os.makedirs(dir_add_new)

    if useMP: # start a multiprocessing.Pool
        p = mp.Pool(mp.cpu_count())

    time_weights = np.zeros(num_Exp)
    for Exp_ID in list_Exp_ID:
        # fname_raw = os.path.join(dir_video, Exp_ID+'.h5')
        # video_raw = np.array(h5py.File(fname_raw, 'r')['mov'])

        # Create the memory mapping file for the masks
        filename_masks = os.path.join(dir_masks, 'Output_Masks_'+Exp_ID+'.mat')
        file_masks = sio.loadmat(filename_masks)
        Masks = file_masks['Masks'].astype('bool')#.transpose([2,1,0])
        # MasksT = Masks.T
        N, Ly, Lx = masks_shape = Masks.shape
        fn_masks = Exp_ID + '_masks.dat'
        fp_masks = np.memmap(fn_masks, dtype='bool', mode='w+', shape=masks_shape)
        fp_masks[:] = Masks[:]

        # Calculate SNR video
        Params_pre = {'gauss_filt_size':50, 'num_median_approx':nframes, \
            'nn':nframes, 'Poisson_filt': 1}
        video_SNR, start = preprocess_video(dir_video, Exp_ID, Params_pre, \
            useSF=True, useTF=False, useSNR=True, prealloc=True)
        video_SNR = gaussian_filter(video_SNR, sigma=(0, 0.5, 0.5), mode='nearest')
        video_SNR = video_SNR[:, :Ly, :Lx]
        T, Ly, Lx = video_shape = video_SNR.shape
        video_dtype = video_SNR.dtype
        num_avg = int(np.maximum(60, np.minimum(90, np.ceil(T * 0.01))))

        # Create memory mapping files for SNR video and masks
        fn_video = Exp_ID+'_video_SNR.dat'
        if os.path.exists(fn_video):
            os.remove(fn_video)
        fp_video = np.memmap(fn_video, dtype=video_dtype, mode='w+', shape=video_shape)
        for tt in range(T):
            fp_video[tt] = video_SNR[tt]#.ravel()

        ##
        traces_raw = traces_from_masks_mmap(fn_video, video_dtype, video_shape, fn_masks, masks_shape, useMP, p)
        ##
        list_weight, list_weight_trace, list_weight_frame = \
            frame_weight_blockwise(fn_video, video_dtype, video_shape, fn_masks, masks_shape, traces_raw, leng, useMP, p)
        ##
        area = Masks.sum(2).sum(1)
        avg_area = np.median(area)
        (masks_added_full, images_added_crop, masks_added_crop, patch_locations, added_frames, added_weights)\
            = find_missing_blockwise(fn_video, video_dtype, video_shape, fn_masks, masks_shape,\
                leng, list_weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split, useMP, p)

        ## Calculate the neighboring neurons
        n_add = added_frames.size
        if n_add > 0:
            list_neighbors = np.zeros(n_add,'object')
            masks_sum = Masks.sum(0)
            for n in range(n_add):
                loc = patch_locations[n,:]
                list_neighbors[n] = (masks_sum[loc[2]:loc[3]+1,loc[0]:loc[1]+1])
            masks_neighbors_crop = np.stack(list_neighbors, 0)
        else:
            masks_neighbors_crop = masks_added_crop.copy()
        time_weights = time.time() - start

        ## Save Masks
        sio.savemat(os.path.join(dir_add_new, Exp_ID+'_added_auto_blockwise.mat'),
            {'added_frames': added_frames, 'added_weights': added_weights, 'masks_added_full': masks_added_full,
            'masks_added_crop': masks_added_crop, 'images_added_crop': images_added_crop, 'patch_locations': patch_locations,
            'time_weights':time_weights ,'masks_neighbors_crop':masks_neighbors_crop}, do_compression=True)

        ##
        fp_video._mmap.close()
        del fp_video
        os.remove(fn_video)
        fp_masks._mmap.close()
        del fp_masks
        os.remove(fn_masks)

    if useMP:
        p.close()
        p.join()
