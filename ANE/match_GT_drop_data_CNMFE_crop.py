data_ind = 3
import numpy as np
import scipy.io as sio
import os
from scipy import sparse
from IoU_2 import IoU_2


if __name__ == "__main__":
    list_data_names = ['blood_vessel_10Hz', 'PFC4_15Hz', 'bma22_epm', 'CaMKII_120_TMT Exposure_5fps']
    list_ID_part = ['_part11', '_part12', '_part21', '_part22']
    # list_avg_radius = [5, 6, 8, 14]
    list_lam = [15, 5, 8, 8]
    r_bg_ratio = 3
    data_name = list_data_names[data_ind]
    list_Exp_ID = [data_name+x for x in list_ID_part]
    d0 = 0.8
    # sub_added = ''
    lam = list_lam[data_ind]

    dir_parent = os.path.join('..','data','data_CNMFE',data_name)
    dir_masks = os.path.join(dir_parent,'GT Masks dropout {}exp(-{})'.format(d0, lam))
    dir_add_new = os.path.join(dir_masks,'add_new_blockwise')

    ##
    for Exp_ID in list_Exp_ID:
        ##
        added_auto_blockwise = sio.loadmat(os.path.join(dir_add_new,Exp_ID+'_added_auto_blockwise.mat'))
        added_frames = added_auto_blockwise['added_frames']
        added_weights = added_auto_blockwise['added_weights']
        masks_added_full = added_auto_blockwise['masks_added_full']
        masks_added_crop = added_auto_blockwise['masks_added_crop']
        images_added_crop = added_auto_blockwise['images_added_crop']
        patch_locations = added_auto_blockwise['patch_locations']
        
        FinalMasks = sio.loadmat(os.path.join(dir_masks,'FinalMasks_'+Exp_ID+'.mat'))
        masks = FinalMasks['FinalMasks'].transpose([2,1,0])
        DroppedMasks = sio.loadmat(os.path.join(dir_masks,'DroppedMasks_'+Exp_ID+'.mat'))
        masks_add_GT = DroppedMasks['DroppedMasks'].transpose([2,1,0])

        ## Find valid neurons
        _, Ly, Lx = masks.shape
        mask_new_full_2 = masks_added_full.reshape((-1, Ly * Lx))
        mask_add_full_2 = masks_add_GT.reshape((-1, Ly * Lx))
        IoU = IoU_2(sparse.csr_matrix(mask_add_full_2),sparse.csr_matrix(mask_new_full_2))
        list_valid = np.any(IoU > 0.3,1)

        ## Calculate the neighboring neurons, and updating masks
        N = len(added_frames)
        list_neighbors = []
        masks_sum = masks.sum(0)
        for n in range(N):
            loc = patch_locations[n,:]
            list_neighbors.append(masks_sum[loc[2]:loc[3]+1,loc[0]:loc[1]+1])
        masks_neighbors_crop = np.stack(list_neighbors, 0)
        ##
        sio.savemat(os.path.join(dir_add_new,Exp_ID+'_added_CNNtrain_blockwise.mat'),\
            {'added_frames':added_frames,'added_weights':added_weights,'list_valid':list_valid,\
            'masks_neighbors_crop':masks_neighbors_crop,'masks_added_crop':masks_added_crop,'images_added_crop':images_added_crop})
