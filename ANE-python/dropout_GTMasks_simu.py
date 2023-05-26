import sys
import scipy.io as sio
import os
import numpy as np
from scipy import special
import h5py


if __name__ == "__main__":
    num_Exp = 10
    list_Exp_ID = ['sim_'+str(x) for x in range(0,num_Exp)]

    r_bg_ratio = 3
    d0 = 0.8

    # Load traces and ROIs
    lam = 8
    name_video = sys.argv[1] # 'lowBG=5e+03,poisson=1'
    dir_parent = os.path.join('../data/data_simulation',name_video)
    dir_video = dir_parent
    dir_masks = os.path.join(dir_parent, 'GT Masks')
    dir_traces_raw = os.path.join(dir_video, 'SUNS_TUnCaT_SF25', 'TUnCaT', 'raw')
    dir_traces_unmix = os.path.join(dir_video, 'SUNS_TUnCaT_SF25', 'TUnCaT', 'alpha= 1.000')
    doesunmix = 1
    dir_add_new = os.path.join(dir_parent, 'GT Masks dropout {}exp(-{})'.format(d0, lam))
    if not os.path.exists(dir_add_new):
        os.makedirs(dir_add_new)

    list_keep = []
    for Exp_ID in list_Exp_ID:
        filename_masks = os.path.join(dir_masks, 'FinalMasks_'+Exp_ID+'.mat')
        try:
            file_masks = sio.loadmat(filename_masks)
            FinalMasks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            FinalMasks = np.array(file_masks['FinalMasks']).astype('bool')
            file_masks.close()

        if doesunmix:
            traces_nmfdemix = sio.loadmat(os.path.join(dir_traces_unmix, Exp_ID+'.mat'))
            traces = traces_nmfdemix['traces_nmfdemix']
            addon = 'unmix '
        else:
            traces_raw = sio.loadmat(os.path.join(dir_traces_raw, Exp_ID+'.mat'))
            traces = (traces_raw['traces'] - traces_raw['bgtraces'])
            addon = 'nounmix '

        # Calculate PSNR
        q12 = np.quantile(traces, [0.25, 0.5], axis=0) 
        sigma = (q12[1]-q12[0])/(np.sqrt(2)*special.erfinv(0.5))
        SNR = (traces - q12[1]) / sigma
        PSNR = SNR.max(0)
        num = len(PSNR)

        # Drop out neurons, with the probability proportional to exp(-PSNR)
        prob = d0*np.exp(- PSNR / lam)
        keep = np.random.rand(num) >= prob
        DroppedMasks = FinalMasks[np.logical_not(keep), :, :].transpose([2,1,0])
        FinalMasks = FinalMasks[keep, :, :].transpose([2,1,0])
        list_keep.append(keep)
        # Save Masks
        sio.savemat(os.path.join(dir_add_new, 'FinalMasks_' + Exp_ID+'.mat'), \
            {'FinalMasks': FinalMasks}, do_compression=True)
        sio.savemat(os.path.join(dir_add_new, 'DroppedMasks_' + Exp_ID+'.mat'), \
            {'DroppedMasks': DroppedMasks}, do_compression=True)

    # Save list_keep
    num_total = np.array([x.size for x in list_keep])
    num_keep = np.array([x.sum() for x in list_keep])
    num_drop = num_total - num_keep
    drop_ratio = num_drop.sum() / num_total.sum()
    sio.savemat(os.path.join(dir_add_new, 'list_keep.mat'), {'list_keep': list_keep,
        'num_total': num_total, 'num_keep': num_keep, 'num_drop': num_drop})
