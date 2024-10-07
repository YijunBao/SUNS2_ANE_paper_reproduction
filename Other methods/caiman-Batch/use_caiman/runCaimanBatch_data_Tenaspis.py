#!/usr/bin/env python

"""
Revised the demo_pipeline.py script to work with Allen dataset

10/19/2018
"""
#%%
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import time
import os
from os import makedirs 
import sys
sys.path.insert(0, 'C:\\Other methods\\CaImAn')
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # "-1", "1"
#os.environ["OPENBLAS_NUM_THREADS"]= "1"
#os.environ["MKL_NUM_THREADS"]="1"
os.environ["CAIMAN_DATA"] = ".\\" #"D:\\ABO\\20 percent bin 5\\caiman-Batch\\" #

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# # config.gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.Session(config = config)

try:
    cv2.setNumThreads(0)
except:
    pass


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.base.rois import mask_to_2d
from caiman.source_extraction.cnmf.cnmf import load_CNMF


# from custom_codes_batch_noCNN import Performance_metrics, Run_caimanBatch, multipleThresh_caimanBatch
from custom_codes_batch import Performance_metrics, Run_caimanBatch, multipleThresh_caimanBatch
import h5py
import keras

# %%
# def main():
if __name__ == "__main__":
    # time.sleep(3600*1.3)
    pass  # For compatibility between running under Spyder and the CLI
    
    #% List of dataID
    Layer = '275'
    task = 'train' # train, test
    rbin = 1

    datadir = 'D:/data_TENASPIS/added_refined_masks/'

    if Layer == '275':
        GTdir = datadir + 'GT Masks\\'
    # 275
        ID = [  'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
                'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']
        if task =='train':
            #% These parameters are optimised
            rval_thr = list(np.arange(0.5,0.95,0.1)) + [0.95, 0.98]
            min_SNR = list(np.arange(2,10))  # signal to noise ratio for accepting a component
            cnn_thr = [1]  # threshold for CNN based classifier
            cnn_lowest = [0] # neurons with cnn probability lower than this value are rejected
            # rval_thr = [0.8, 0.9, 0.95, 0.98]
            # min_SNR = [2, 3, 4, 5]  # signal to noise ratio for accepting a component
            # cnn_thr = [0.8, 0.9, 0.95, 0.98]  # threshold for CNN based classifier
            # cnn_lowest = [0, 0.1, 0.2, 0.3, 0.4] # neurons with cnn probability lower than this value are rejected
        else:
            rval_thr = [0.98,0.98,0.98,0.98, 0.95,0.98,0.98,0.98]
            min_SNR = [3]*8 # [8,6,8,8,8,8,8,8,8,8]
            cnn_thr = [1]*8 # [0.9]*10
            cnn_lowest = [0]*8 # 
            # rval_thr = [0.95]*8
            # min_SNR = [3]*8 # [8,6,8,8,8,8,8,8,8,8]
            # cnn_thr = [0.8]*8 # [0.9]*10
            # cnn_lowest = [0.4]*8 # 
    else:
        # 175
        GTdir = datadir + 'GT Masks\\'
        ID = [  'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
                'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']
        rval_thr = [0.9]*10
        min_SNR = [4]*10
        cnn_thr = [0.9]*10
        cnn_lowest = [0.4]*10 # ophys_experiment_

    hdf5dir = datadir+'caiman-Batch\\hdf5\\'
    if not os.path.exists(hdf5dir):
        makedirs(hdf5dir)
        
    if task == 'train':
        savedir = datadir+'caiman-Batch\\train\\'
        if not os.path.exists(savedir):
            makedirs(savedir)
    else:
        savedir = datadir+'caiman-Batch\\'+Layer+'\\'
        if not os.path.exists(savedir):
            makedirs(savedir)
    
    GTName = 'FinalMasks_'
    # GT mask for all data
    GTMask = {}
    for expID in ID:
        try:
            f = h5py.File(GTdir+GTName+expID+'.mat','r')
            m = np.array(f['FinalMasks'])
        except:
            f = sio.loadmat(GTdir+GTName+expID+'.mat')
            m = np.array(f['FinalMasks']).transpose([2,1,0])
        # Uncomment below if (512,512) movies will be processed by caiman
        # m = np.pad(m,((0,0),(12,13),(12,13)),'constant')
        GTMask[expID] = mask_to_2d(m).astype('float')   
       
    extension = '.mmap'
        
#%% First setup some parameters for data and motion correction
    # dataset dependent parameters
    is_patches = True       # flag for processing in patches or not
    fr = 20/rbin            # approximate frame rate of data
    decay_time = 1       # length of transient

    if is_patches:          # PROCESS IN PATCHES AND THEN COMBINE
        rf = 30             # half size of each patch
        stride = 10         # overlap between patches
        K = 20              # number of components in each patch
    else:                   # PROCESS THE WHOLE FOV AT ONCE
        rf = None           # setting these parameters to None
        stride = None       # will run CNMF on the whole FOV
        K = 500             # number of neurons expected (in the whole FOV)

    gSig = [9,9]            # expected half size of neurons
    merge_thresh = 0.80     # merging threshold, max correlation allowed
    p = 1                   # order of the autoregressive system
    gnb = 2                 # global background order


    method_init = 'greedy_roi'
    ssub = 1                     # spatial subsampling during initialization
    tsub = 1                     # temporal subsampling during intialization

#%%
# if __name__ == "__main__":
    AllFiles = glob.glob(datadir+"*"+extension)  
        
    # CV = 0  # Which fold of cross-validation? Only applicable for "train" mode
            # 0-9 are for 1-fold cross-validation. CV=10 will use all layer 275 data for optimization
    # if Layer == '175':
    list_CV = [len(ID)]
    # else:
    #     list_CV = list(range(len(ID)))

    for CV in list_CV: # [10]: # range(0,10): #
        cnn_name = '' # '_Allen_'+str(CV)
        if task == 'train': 
            if CV<len(ID):
                ID_loop = [ elem for elem in ID if elem != ID[CV]]
            else:
                ID_loop = ID
            F1All = np.zeros([len(ID_loop),len(rval_thr),len(min_SNR),len(cnn_thr),len(cnn_lowest)])
        else:
            if CV < len(ID):
                ID_loop = [ID[CV]]
            else:
                ID_loop = ID
        
        for cnt,IDTrain in enumerate(ID_loop): #[::-1]
            # if cnt<=5:
            #     continue
            #% Select file(s) to be processed (download if not present)
            expID = IDTrain
            # if task == 'test':
            #     cnn_name = '_Allen_' + str(cnt) # ''# 

            fnames = [i for i in AllFiles if i.find(expID)!=-1]#glob.glob(datadir + expID + extension)
            
            saveInfo = {'Dir': savedir,
                    'Name': IDTrain + '.mat'}
            if extension=='.mmap':
                fnames=fnames[0]
            print(fnames)
        # % start a cluster
            # if cnt==5:
            # n_processes = 20
            # else:
            n_processes = None
            c, dview, n_processes =\
                cm.cluster.setup_cluster(backend='local', n_processes=n_processes,
                                    single_thread=False)
            params_dict = {'fnames': fnames,
                        'fr': fr,
                        'decay_time': decay_time,
                        'rf': rf,
                        'stride': stride,
                        'K': K,
                        'gSig': gSig,
                        'merge_thr': merge_thresh,
                        'p': p,
                        'nb': gnb,
                        'method_init': method_init,
                        'rolling_sum': True, 
                        'n_processes': n_processes,
                        'only_init': True,
                        'ssub': ssub,
                        'tsub': tsub,
                        'gSig_range': 2,
                        'check_nan':False}

            opts = params.CNMFParams(params_dict=params_dict)
            # % Now RUN CaImAn Batch (CNMF)
            if task == 'train':
                mat_name = savedir+'CV'+str(CV)+'\\'+IDTrain+'.mat'
                if False: # os.path.exists(mat_name):
                    previous_result = sio.loadmat(mat_name)
                    f = previous_result['f1']
                else:
                    hdf5_name = hdf5dir+IDTrain+'.hdf5'
                    # % load memory mapped file
                    Yr, dims, T = cm.load_memmap(fnames) #This is for the case where the mmap files have been saved previously with a specific name   
                    images = np.reshape(Yr.T, [T] + list(dims), order='F')
                    images.max()
                        
                    if not os.path.exists(hdf5_name):
                        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
                        start_fit=time.time()
                        cnm = cnm.fit_file()   # images 
                        # % RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
                        cnm.params.set('temporal', {'p': 1})
                        cnm2 = cnm.refit(images, dview=dview)

                        print(time.time()-start_fit)
                        #% save to hdf5 and reload for each combination of parameters
                        cnm2.save(hdf5_name)
                    # else:
                    #     cnm = load_CNMF(hdf5_name)
                                    
                    save_results = len(rval_thr)*len(min_SNR)*len(cnn_thr)*len(cnn_lowest)==1 # False
                    rec,prec,f, ptime = multipleThresh_caimanBatch(Yr,images, hdf5_name, dview,rval_thr,min_SNR, cnn_thr, cnn_lowest,\
                        dims,IDTrain,GTMask,save_results,saveInfo,cnn_name) # _Allen_'+str(CV))
                    if not os.path.exists(savedir+'CV'+str(CV)):
                        makedirs(savedir+'CV'+str(CV))
                    sio.savemat(mat_name,{'f1':f, 'recall':rec, 'precision':prec, 'processing_time':ptime})
                F1All[cnt,:] = f
    # %%
            else:
                save_results = True
                if Layer =='275':
                    Run_caimanBatch(n_processes,opts, dview,fnames,rval_thr[CV],min_SNR[CV], cnn_thr[CV],cnn_lowest[CV],\
                        IDTrain,GTMask,save_results,saveInfo,hdf5dir,cnn_name) # '_Allen_'+str(cnt)) # [0]
                else:
                    Run_caimanBatch(n_processes,opts, dview,fnames,rval_thr[cnt],min_SNR[cnt], cnn_thr[cnt],cnn_lowest[cnt],\
                        IDTrain,GTMask,save_results,saveInfo,hdf5dir,'_Allen_10') # [0]
                
                sio.savemat(savedir+'Parameters.mat',{'rval_thr': rval_thr, 'min_SNR':min_SNR,'cnn_thr':cnn_thr, \
                    'cnn_lowest': cnn_lowest, 'rf':rf, 'K':K, 'decay_time': decay_time})
            
            #% STOP CLUSTER and clean up log files
            if task == 'train':
                keras.backend.clear_session()
            cm.stop_server(dview=dview)
            log_files = glob.glob('*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)
            
    #%% Find best parameters. The last one shows paramters to be used for 175 um data.
        if task == 'train':
    #        rvalThr = np.zeros(len(ID)+1)
    #        minSNR = np.zeros(len(ID)+1)
    #        cnnLowest = np.zeros(len(ID)+1)
    #        cnnThr = np.zeros(len(ID)+1)
    #        D = list(range(0,len(ID)))

    #        for L in range(0,len(ID)+1):
    #            if L==len(ID_loop):
            F1mean = np.mean(F1All,axis=0)
    #            else:
    #                d = [x for x in D if x!=L]
    #                F1mean = np.mean(F1All[d,:],axis=0)  
            ind = np.argmax(F1mean.flatten())
            ind = np.unravel_index(ind,(len(rval_thr),len(min_SNR),len(cnn_thr),len(cnn_lowest)))
            
            # save best thresholds
            rvalThr = rval_thr[ind[0]]
            minSNR = min_SNR[ind[1]]
            cnnThr = cnn_thr[ind[2]] 
            cnnLowest = cnn_lowest[ind[3]] 
            
            print('Best rval_thr: %f , min_SNR: %f, , cnn_thr: %f, , cnn_lowest: %f:'%(rvalThr,minSNR,cnnThr,cnnLowest))
            sio.savemat(savedir+'Params'+str(CV)+'.mat',{'rval_thr': rvalThr, 'min_SNR':minSNR,'cnn_thr':cnnThr, 'cnn_lowest': cnnLowest})        
            sio.savemat(savedir+'F1All '+str(CV)+'.mat',{'F1All': F1All, 'rval_thr': rval_thr, 'min_SNR':min_SNR,'cnn_thr':cnn_thr, 'cnn_lowest': cnn_lowest})        
        
    # %%
    # This is to mask the differences between running this demo in Spyder
    # versus from the CLI
    # if __name__ == "__main__":
    #     main()
