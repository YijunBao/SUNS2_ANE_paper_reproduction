import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
import h5py

# if (sys.version_info.major+sys.version_info.minor/10)>=3.8
from multiprocessing.shared_memory import SharedMemory
from traces_from_masks_numba_neighbors import traces_bgtraces_from_masks_numba_neighbors
from traces_from_masks_mp_shm_neighbors import traces_bgtraces_from_masks_shm_neighbors
from use_nmfunmix_mp_diag_v1_shm_MSE_novideo import use_nmfunmix


if __name__ == '__main__':
    # sys.argv = ['py', 'Raw', '1', '0', '1', '1']
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    video_type = sys.argv[1] # 'Raw' # 'SNR' # 
    # dir_video = 'D:\\ABO\\20 percent 200' 
    dir_video = '..\\data\\ABO\\'

    Qclip = 0
    epsilon = 0
    nbin = int(sys.argv[2]) # 1 # 
    bin_option = 'downsample' # 'mean' # 'sum' # 
    if nbin == 1:
        addon = ''
    else:
        addon = '_'+bin_option +str(nbin)

    th_pertmin = float(sys.argv[4]) # 1 #
    if th_pertmin < 1:
        addon += '_th_pertmin={}'.format(sys.argv[4])
    
    th_residual = float(sys.argv[3]) # 0
    if th_residual > 0:
        addon += '_residual0={}'.format(sys.argv[3])

    flexible_alpha = bool(int(sys.argv[5])) # True # 
    if flexible_alpha:
        # addon += ''
        list_alpha = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]
        # addon += '_1000'
        # list_alpha = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
    else:
        addon += '_fixed_alpha'
        list_alpha = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
    Table_time = np.zeros((len(list_Exp_ID), len(list_alpha)+1))

    # Load video and FinalMasks
    if video_type == 'SNR':
        varname = 'network_input'
        dir_video_SNR = os.path.join(dir_video, 'SNR video')
    else:
        varname = 'mov'
        dir_video_SNR = dir_video
    dir_masks = os.path.join(dir_video, 'SUNS_complete Masks')
    dir_traces = os.path.join(dir_video, 'traces_SUNS_complete+ours_'+video_type + addon)
    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 
    dir_trace_raw = os.path.join(dir_traces, "raw")
    if not os.path.exists(dir_trace_raw):
        os.makedirs(dir_trace_raw)        
    
    for (ind_Exp, Exp_ID) in enumerate(list_Exp_ID):
        print(Exp_ID)
        start = time.time()
        filename_video = os.path.join(dir_video_SNR, Exp_ID + '.h5')
        file_video = h5py.File(filename_video, 'r')
        (T, Lx, Ly) = video_shape = file_video[varname].shape
        video_dtype = file_video[varname].dtype
        nbytes_video = int(video_dtype.itemsize * file_video[varname].size)
        shm_video = SharedMemory(create=True, size=nbytes_video)
        video = np.frombuffer(shm_video.buf, dtype = file_video[varname].dtype)
        video[:] = file_video[varname][()].ravel()
        video = video.reshape(file_video[varname].shape)
        file_video.close()

        filename_masks = os.path.join(dir_masks, 'FinalMasks_' + Exp_ID + '.mat')
        try:
            file_masks = loadmat(filename_masks)
            Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            Masks = np.array(file_masks['FinalMasks']).astype('bool')
            file_masks.close()
        (ncells, Lxm, Lym) = Masks.shape
        if Lxm < Lx:
            Masks = np.pad(Masks,((0,0),(0,Lx-Lxm),(0,0)),'constant', constant_values=0)
        elif Lxm > Lx:
            Masks = Masks[:,:Lx,:]
        if Lym < Ly:
            Masks = np.pad(Masks,((0,0),(0,0),(0,Ly-Lym)),'constant', constant_values=0)
        elif Lym > Ly:
            Masks = Masks[:,:,:Ly]
        masks_shape = Masks.shape
        shm_masks = SharedMemory(create=True, size=Masks.nbytes)
        FinalMasks = np.frombuffer(shm_masks.buf, dtype = 'bool')
        FinalMasks[:] = Masks.ravel()
        FinalMasks = FinalMasks.reshape(Masks.shape)
        del Masks
        finish = time.time()
        print(finish - start)

        # Calculate traces and background traces
        start = time.time()
        if FinalMasks.sum()*T >= 7e7: # Use multiprocessing is faster for large videos
            (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_shm_neighbors(
                shm_video, video_dtype, video_shape, shm_masks, masks_shape, FinalMasks)
        else: # Use numba is faster for small videos
            (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_numba_neighbors(
                video, FinalMasks)

        finish = time.time()
        print('trace calculation time: {} s'.format(finish - start))
        Table_time[ind_Exp, -1] = finish-start

        # Save the raw traces into a ".mat" file under folder "dir_trace_raw".
        savemat(os.path.join(dir_trace_raw, Exp_ID+".mat"), {"traces": traces, "bgtraces": bgtraces}) # , "outtraces": outtraces

        for (ind_alpha, alpha) in enumerate(list_alpha):
            print(Exp_ID, 'alpha =', alpha)
            # Do NMF unmixing
            start = time.time()
            traces_nmfdemix, list_mixout, list_MSE, list_final_alpha, list_n_iter = use_nmfunmix(
                traces, bgtraces, outtraces, list_neighbors, [alpha], Qclip, \
                th_pertmin, epsilon, th_residual, nbin, bin_option, flexible_alpha)
            finish = time.time()
            print('unmixing time: {} s'.format(finish - start))
            Table_time[ind_Exp, ind_alpha] = finish-start

            # Save the unmixed traces into a ".mat" file under folder "dir_trace_unmix".
            dir_trace_unmix = os.path.join(dir_traces, "alpha={:6.3f}".format(alpha))
            if not os.path.exists(dir_trace_unmix):
                os.makedirs(dir_trace_unmix)        
            savemat(os.path.join(dir_trace_unmix, Exp_ID+".mat"), {"traces_nmfdemix": traces_nmfdemix,\
                "list_mixout":list_mixout, "list_MSE":list_MSE, "list_final_alpha":list_final_alpha, "list_n_iter":list_n_iter})

        shm_video.close()
        shm_video.unlink()
        shm_masks.close()
        shm_masks.unlink()

        savemat(os.path.join(dir_traces, "Table_time.mat"), {"Table_time": Table_time, 'list_alpha': list_alpha})

