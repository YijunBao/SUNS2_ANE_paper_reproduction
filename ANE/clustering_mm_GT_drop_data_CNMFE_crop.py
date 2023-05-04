import scipy.io as sio
import os
import numpy as np
import h5py
import sys
# from skimage import filters
from scipy.ndimage import gaussian_filter
from scipy import sparse
from IoU_2 import IoU_2

sys.path.insert(1, '../SUNS2') # the path containing "suns" folder
from suns.PreProcessing.preprocessing_functions import preprocess_video
from frame_weight_blockwise_mm import frame_weight_blockwise_mm
from find_missing_blockwise_mm import find_missing_blockwise_mm
from traces_from_masks_mmap import traces_from_masks_mmap
data_ind = 3


if __name__ == "__main__":
    list_data_names = ['blood_vessel_10Hz', 'PFC4_15Hz', 'bma22_epm', 'CaMKII_120_TMT Exposure_5fps']
    list_ID_part = ['_part11', '_part12', '_part21', '_part22']
    data_name = list_data_names[data_ind]
    list_Exp_ID = [data_name+x for x in list_ID_part]
    num_Exp = len(list_Exp_ID)
    rate_hz = [10, 15, 7.5, 5]
    list_nframes = [6000, 9000, 9000, 1500]

    list_avg_radius = [5, 6, 8, 14]
    list_lam = [15, 5, 8, 8]
    r_bg_ratio = 3
    leng = r_bg_ratio * list_avg_radius[data_ind]
    th_IoU_split = 0.5
    thj_inclass = 0.4
    thj = 0.7

    d0 = 0.8
    lam = list_lam[data_ind]
    dir_parent = os.path.join('..', 'data', 'data_CNMFE', data_name)
    dir_video = dir_parent
    dir_masks = os.path.join(dir_parent, 'GT Masks dropout {}exp(-{})'.format(d0, lam))
    dir_add_new = os.path.join(dir_masks, 'add_new_blockwise')
    if not os.path.exists(dir_add_new):
        os.makedirs(dir_add_new)

    time_weights = np.zeros(num_Exp)
    for Exp_ID in range(list_Exp_ID):
        # fname_raw = os.path.join(dir_video, Exp_ID+'.h5')
        # video_raw = np.array(h5py.File(fname_raw, 'r')['mov'])

        Params_pre = {'gauss_filt_size':50, 'num_median_approx':1000, \
            'nn':list_nframes[data_ind], 'Poisson_filt': 1}
        video_SNR, start = preprocess_video(dir_video, Exp_ID, Params_pre, \
            useSF=True, useTF=False, useSNR=True, prealloc=True)
        video_SNR = gaussian_filter(video_SNR, sigma=(0, 0.5, 0.5), mode='nearest')
        T, Ly, Lx = video_shape = video_SNR.shape
        video_dtype = video_SNR.dtype
        npatchx = np.ceil(Lx / leng) - 1
        npatchy = np.ceil(Ly / leng) - 1
        num_avg = np.maximum(60, np.minimum(90, np.ceil(T * 0.01)))

        # Create memory mapping files for SNR video and masks
        fn_video = Exp_ID+'video_SNR.dat'
        if os.path.exists(fn_video):
            os.remove(fn_video)
        fp_video = np.memmap(fn_video, dtype=video_dtype, mode='w+', shape=video_shape)
        for tt in range(T):
            fp_video[tt] = video_SNR[tt].ravel()

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
        masks_shape = Masks.shape
        fn_masks = Exp_ID + 'masks.dat'
        fp_masks = np.memmap(fn_masks, dtype='bool', mode='w+', shape=masks_shape)
        fp_masks[:] = Masks[:]

        ##
        traces_raw = traces_from_masks_mmap(fn_video, video_dtype, video_shape, fn_masks, masks_shape)
        ##
        list_weight, list_weight_trace, list_weight_frame = \
            frame_weight_blockwise_mm(fn_video, video_dtype, video_shape, fn_masks, masks_shape, traces_raw, leng)
        ##
        area = Masks.sum(2).sum(1)
        avg_area = np.median(area)
        list_added_full = np.zeros((npatchy, npatchx), 'object')
        list_added_crop = np.zeros((npatchy, npatchx), 'object')
        list_added_images_crop = np.zeros((npatchy, npatchx), 'object')
        list_added_frames = np.zeros((npatchy, npatchx), 'object')
        list_added_weights = np.zeros((npatchy, npatchx), 'object')
        list_locations = np.zeros((npatchy, npatchx), 'object')
        ##
        for iy in range(npatchy):
            for ix in range(npatchx):
                xmin = np.minimum(Lx - 2 * leng + 1, (ix - 1) * leng + 1)-1
                xmax = np.minimum(Lx, (ix + 1) * leng)-1
                ymin = np.minimum(Ly - 2 * leng + 1, (iy - 1) * leng + 1)-1
                ymax = np.minimum(Ly, (iy + 1) * leng)-1
                weight = list_weight[iy, ix]
                image_new_crop, mask_new_crop, mask_new_full, select_frames_class, select_weight_calss = \
                    find_missing_blockwise_mm(fn_video, video_dtype, video_shape, fn_masks, masks_shape,\
                        xmin, xmax, ymin, ymax, weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split)
                list_added_images_crop[iy, ix] = image_new_crop
                list_added_crop[iy, ix] = mask_new_crop
                list_added_full[iy, ix] = mask_new_full
                list_added_frames[iy, ix] = select_frames_class
                list_added_weights[iy, ix] = select_weight_calss
                n_class = image_new_crop.shape[0]
                list_locations[iy, ix] = np.repeat([[xmin, xmax, ymin, ymax]], n_class, 0)
        
        # Save Masks
        masks_added_full = np.stack(list_added_full.ravel(), 0)
        images_added_crop = np.stack(list_added_images_crop.ravel(), 0)
        masks_added_crop = np.stack(list_added_crop.ravel(), 0)
        patch_locations = np.concatenate(list_locations.ravel(), 0)
        added_frames = np.concatenate(list_added_frames.ravel())
        added_weights = np.concatenate(list_added_weights.ravel())
        # sio.savemat(os.path.join(dir_add_new, Exp_ID+'_added_auto_blockwise.mat'),
        #     {'added_frames': added_frames, 'added_weights': added_weights, 'masks_added_full': masks_added_full,
        #     'masks_added_crop': masks_added_crop, 'images_added_crop': images_added_crop, 'patch_locations': patch_locations})


        # %% Match putative neurons to dropped neurons. 
        # FinalMasks = sio.loadmat(os.path.join(dir_masks,'FinalMasks_'+Exp_ID+'.mat'))
        # masks = FinalMasks['FinalMasks'].transpose([2,1,0])
        masks = Masks
        DroppedMasks = sio.loadmat(os.path.join(dir_masks,'DroppedMasks_'+Exp_ID+'.mat'))
        masks_add_GT = DroppedMasks['DroppedMasks'].transpose([2,1,0])

        ## Find valid neurons
        # _, Ly, Lx = masks.shape
        mask_new_full_2 = masks_added_full.reshape((-1, Ly * Lx))
        mask_add_full_2 = masks_add_GT.reshape((-1, Ly * Lx))
        IoU = IoU_2(sparse.csr_matrix(mask_add_full_2),sparse.csr_matrix(mask_new_full_2))
        list_valid = np.any(IoU > 0.3,1)

        ## Calculate the neighboring neurons, and updating masks
        list_neighbors = []
        masks_sum = masks.sum(0)
        for n in range(added_frames.size):
            loc = patch_locations[n,:]
            list_neighbors.append(masks_sum[loc[2]:loc[3]+1,loc[0]:loc[1]+1])
        masks_neighbors_crop = np.stack(list_neighbors, 0)
        ##
        sio.savemat(os.path.join(dir_add_new,Exp_ID+'_added_CNNtrain_blockwise.mat'),\
            {'added_frames':added_frames,'added_weights':added_weights,'list_valid':list_valid,\
            'masks_neighbors_crop':masks_neighbors_crop,'masks_added_crop':masks_added_crop,'images_added_crop':images_added_crop})

        ##
        fp_video._mmap.close()
        del fp_video
        os.remove(fn_video)
        fp_masks._mmap.close()
        del fp_masks
        os.remove(fn_masks)
