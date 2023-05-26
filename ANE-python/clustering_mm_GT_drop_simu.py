import scipy.io as sio
import os
import numpy as np
import h5py
import sys
# from skimage import filters
from scipy.ndimage import gaussian_filter
from scipy import sparse
from IoU_2 import IoU_2
import multiprocessing as mp

sys.path.insert(1, '../SUNS2') # the path containing "suns" folder
from suns.PreProcessing.preprocessing_functions import preprocess_video
from frame_weight_blockwise_mm import frame_weight_blockwise
from find_missing_blockwise_mm import find_missing_blockwise
from traces_from_masks_mmap import traces_from_masks_mmap


if __name__ == "__main__":
    name_video = sys.argv[1] # 'lowBG=5e+03,poisson=1' # 

    num_Exp = 10
    list_Exp_ID = ['sim_'+str(x) for x in range(0,num_Exp)]
    nframes = 2000

    avg_radius = 6
    r_bg_ratio = 3
    leng = r_bg_ratio * avg_radius
    th_IoU_split = 0.5
    thj_inclass = 0.4
    thj = 0.7
    useMP = True

    d0 = 0.8
    lam = 8
    dir_parent = os.path.join('../data/data_simulation',name_video)
    dir_video = dir_parent
    dir_masks = os.path.join(dir_parent, 'GT Masks dropout {}exp(-{})'.format(d0, lam))
    dir_add_new = os.path.join(dir_masks, 'add_new_blockwise')
    if not os.path.exists(dir_add_new):
        os.makedirs(dir_add_new)

    if useMP: # start a multiprocessing.Pool
        p = mp.Pool(mp.cpu_count())

    time_weights = np.zeros(num_Exp)
    for Exp_ID in list_Exp_ID:
        # fname_raw = os.path.join(dir_video, Exp_ID+'.h5')
        # video_raw = np.array(h5py.File(fname_raw, 'r')['mov'])

        # Create the memory mapping file for the masks
        filename_masks = os.path.join(dir_masks, 'FinalMasks_'+Exp_ID+'.mat')
        try:
            file_masks = sio.loadmat(filename_masks)
            Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            Masks = np.array(file_masks['FinalMasks']).astype('bool')
            file_masks.close()
        # MasksT = Masks.T
        N, Ly, Lx = masks_shape = Masks.shape
        fn_masks = Exp_ID + '__masks.dat'
        fp_masks = np.memmap(fn_masks, dtype='bool', mode='w+', shape=masks_shape)
        fp_masks[:] = Masks[:]

        # Calculate SNR video
        Params_pre = {'gauss_filt_size':50, 'num_median_approx':1000, \
            'nn':nframes, 'Poisson_filt': 1}
        video_SNR, start = preprocess_video(dir_video, Exp_ID, Params_pre, \
            useSF=True, useTF=False, useSNR=True, prealloc=True)
        video_SNR = gaussian_filter(video_SNR, sigma=(0, 0.5, 0.5), mode='nearest')
        video_SNR = video_SNR[:, :Ly, :Lx]
        T, Ly, Lx = video_shape = video_SNR.shape
        video_dtype = video_SNR.dtype
        num_avg = int(np.maximum(60, np.minimum(90, np.ceil(T * 0.01))))

        # Create memory mapping files for SNR video and masks
        fn_video = Exp_ID+'__video_SNR.dat'
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

        sio.savemat(os.path.join(dir_add_new, Exp_ID+'_added_auto_blockwise.mat'),
            {'added_frames': added_frames, 'added_weights': added_weights, 'masks_added_full': masks_added_full,
            'masks_added_crop': masks_added_crop, 'images_added_crop': images_added_crop, 
            'patch_locations': patch_locations}, do_compression=True)


        # %% Match putative neurons to dropped neurons. 
        # FinalMasks = sio.loadmat(os.path.join(dir_masks,'FinalMasks_'+Exp_ID+'.mat'))
        # Masks = FinalMasks['FinalMasks'].transpose([2,1,0]).astype('bool')
        DroppedMasks = sio.loadmat(os.path.join(dir_masks,'DroppedMasks_'+Exp_ID+'.mat'))
        masks_add_GT = DroppedMasks['DroppedMasks'].transpose([2,1,0]).astype('bool')

        ## Find valid neurons
        # _, Ly, Lx = masks.shape
        mask_new_full_2 = masks_added_full.reshape((-1, Ly * Lx))
        mask_add_full_2 = masks_add_GT.reshape((-1, Ly * Lx))
        IoU = IoU_2(sparse.csr_matrix(mask_add_full_2),sparse.csr_matrix(mask_new_full_2))
        list_valid = np.any(IoU > 0.3,0)

        ## Calculate the neighboring neurons
        list_neighbors = []
        masks_sum = Masks.sum(0)
        for n in range(added_frames.size):
            loc = patch_locations[n,:]
            list_neighbors.append(masks_sum[loc[2]:loc[3]+1,loc[0]:loc[1]+1])
        masks_neighbors_crop = np.stack(list_neighbors, 0)

        ##
        sio.savemat(os.path.join(dir_add_new,Exp_ID+'_added_CNNtrain_blockwise.mat'),\
            {'added_frames':added_frames,'added_weights':added_weights,'list_valid':list_valid,\
            'masks_neighbors_crop':masks_neighbors_crop,'masks_added_crop':masks_added_crop,
            'images_added_crop':images_added_crop}, do_compression=True)

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
