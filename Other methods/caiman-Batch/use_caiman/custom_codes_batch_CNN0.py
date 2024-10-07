#!/usr/bin/env python

"""
Revised the demo_pipeline.py script to work with Allen dataset

10/19/2018
"""

import cv2
import glob
import logging
from time import time
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
try:
    cv2.setNumThreads(0)
except:
    pass

import sys
sys.path.insert(0, 'C:\\Other methods\\CaImAn')

import caiman as cm
import copy 
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.base.rois import mask_to_2d
from caiman.base.rois import register_ROIs
from caiman.source_extraction.cnmf.cnmf import load_CNMF

import h5py
from caiman.paths import caiman_datadir
import os.path
from caiman import mmapping
#%%

def multipleThresh_caimanBatch(Yr,images, hdf5_name, dview,rval_thr,min_SNR, cnn_thr, cnn_lowest,dims,ID,GTMask,save_results,saveInfo,cnt):
    
    recall = np.zeros((len(rval_thr),len(min_SNR),len(cnn_thr),len(cnn_lowest)))
    precision = np.zeros((len(rval_thr),len(min_SNR),len(cnn_thr),len(cnn_lowest)))
    f1 = np.zeros((len(rval_thr),len(min_SNR),len(cnn_thr),len(cnn_lowest))) 
    time_evaluate = np.zeros((len(rval_thr),len(min_SNR),len(cnn_thr),len(cnn_lowest)))
    time_select = np.zeros((len(rval_thr),len(min_SNR),len(cnn_thr),len(cnn_lowest)))
    time_performance = np.zeros((len(rval_thr),len(min_SNR),len(cnn_thr),len(cnn_lowest))) 
    
    # % COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    # cnm2 = load_CNMF('D:\\ABO\\20 percent bin 30\\caiman-Batch\\train\\'+ID+'.hdf5') # bin 5
    cnm2 = load_CNMF(hdf5_name)
    
    for th in range(len(rval_thr)):
        for s in range(len(min_SNR)):  
            for t in range(len(cnn_thr)):
                for l in range(len(cnn_lowest)):
                    cnm3 = copy.deepcopy(cnm2)
                    cnm3.params.set('quality', {#'decay_time':0.5,
                                   'min_SNR': min_SNR[s],
                                   'rval_thr': rval_thr[th],
                                   'use_cnn': True,
                                   'min_cnn_thr': cnn_thr[t],
                                   'cnn_lowest': cnn_lowest[l]})
                    start = time()
                    model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model'+cnt)
                    # model_name = r'D:\ABO\20 percent\caiman-CNN\cnn_model_Allen_10'
                    # model_name = os.path.join(os.path.split(saveInfo['Dir'])[0], 'caiman-CNN', 'cnn_model'+cnt)
                    cnm3.estimates.evaluate_components(images, cnm3.params, dview=dview, model_name=model_name)
                    stop_evaluate = time()
                    #% update object with selected components
                    cnm3.estimates.select_components(use_object=True)
                    stop_select = time()
                    performance = Performance_metrics(GTMask[ID],cnm3.estimates.A,dims)
                    stop_performance = time()

                    recall[th, s, t, l] = performance['recall']
                    precision[th, s, t, l] = performance['precision']
                    f1[th, s, t, l] = performance['f1_score']  
                    time_evaluate[th, s, t, l] = stop_evaluate - start
                    time_select[th, s, t, l] = stop_select - stop_evaluate
                    time_performance[th, s, t, l] = stop_performance - stop_select
                    print({'time_evaluate':time_evaluate[th, s, t, l], 'time_select':time_select[th, s, t, l], 'time_performance':time_performance[th, s, t, l]})

    
    #% save masks to mat file
    processing_time = {'time_evaluate':time_evaluate, 'time_select':time_select, 'time_performance':time_performance}
    if save_results:
        sio.savemat(saveInfo['Dir']+saveInfo['Name'],mdict={'Ab': (cnm3.estimates.A).toarray()},oned_as='column', do_compression=True)
         
    return recall, precision, f1, processing_time


def Run_caimanBatch(n_processes,opts, dview, name,rval_thr,min_SNR, cnn_thr, cnn_lowest,ID,GTMask,save_results,saveInfo,hdf5dir,cnt):
    hdf5_name = hdf5dir+ID+'.hdf5'
    t1 = time()
    if False: # os.path.exists(hdf5_name): # 
        start = time()
        cnm2 = load_CNMF(hdf5_name)
        # % load memory mapped file
        if name[-4:]=='mmap':
            Yr, dims, T = cm.load_memmap(name)  
        else:
            fname_new = mmapping.save_memmap([name], base_name=ID+'_', order='C')
            Yr, dims, T = cm.load_memmap(fname_new)  
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        images.max()

    else:
        #% Now RUN CaImAn Batch (CNMF)
        Yr, dims, T = cm.load_memmap(name) # cnm.mmap_file
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        images.max()
        start = time()
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        cnm = cnm.fit_file()    
        # % RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
        cnm.params.set('temporal', {'p': 1})
        cnm2 = cnm.refit(images, dview=dview)
    
    # % COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    
    cnm2.params.set('quality', {#'decay_time':0.5,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'use_cnn': True,
                   'min_cnn_thr': cnn_thr,
                   'cnn_lowest': cnn_lowest})
    model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model'+cnt)
    # model_name = r'D:\ABO\20 percent\caiman-CNN\cnn_model_Allen_10'
    # model_name = os.path.join(os.path.split(saveInfo['Dir'])[0], 'caiman-CNN', 'cnn_model'+cnt)
    print(model_name)
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview,model_name=model_name)

    #% update object with selected components
    cnm2.estimates.select_components(use_object=True)
    tottime = time()-start   
    # performance = Performance_metrics(GTMask[ID],cnm2.estimates.A,dims)
    # f1 = performance['f1_score']  

    #% save masks to mat file
    if save_results:
        sio.savemat(saveInfo['Dir']+saveInfo['Name'],mdict={'Ab': (cnm2.estimates.A).toarray(),\
            'C': cnm2.estimates.C,'YrA': cnm2.estimates.YrA,'SNR': cnm2.estimates.SNR_comp,'ProcessTime':tottime},\
            oned_as='column', do_compression=True)

    # return f1


def Run_caimanBatch_multiCNN(n_processes,opts, dview, name,rval_thr,min_SNR, cnn_thr, cnn_lowest,ID,GTMask,save_results,saveInfo,hdf5dir,list_cnn_name):
    hdf5_name = hdf5dir+ID+'.hdf5'
    t1 = time()
    if os.path.exists(hdf5_name): # False: # 
        # % load memory mapped file
        if name[-4:]=='mmap':
            Yr, dims, T = cm.load_memmap(name)  
        else:
            fname_new = mmapping.save_memmap([name], base_name=ID+'_', order='C')
            Yr, dims, T = cm.load_memmap(fname_new)  
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        images.max()
        start = time()
        cnm2 = load_CNMF(hdf5_name)
        time_CNMF = time()-start

    else:
        #% Now RUN CaImAn Batch (CNMF)
        Yr, dims, T = cm.load_memmap(name) # cnm.mmap_file
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        images.max()
        start = time()
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        cnm = cnm.fit_file()    
        # % RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
        cnm.params.set('temporal', {'p': 1})
        cnm2 = cnm.refit(images, dview=dview)
        time_CNMF = time()-start
    
    # % COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    
    for (CV,cnn_name) in enumerate(list_cnn_name):
        # if saveInfo['Name'][:-4] not in cnn_name:
        # if CV>0:
        #     continue
        cnm3 = copy.deepcopy(cnm2)
        cnm3.params.set('quality', {#'decay_time':0.5,
                   'min_SNR': min_SNR[CV],
                   'rval_thr': rval_thr[CV],
                   'use_cnn': True,
                   'min_cnn_thr': cnn_thr[CV],
                   'cnn_lowest': cnn_lowest[CV]})
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model'+cnn_name)
        # model_name = r'D:\ABO\20 percent\caiman-CNN\cnn_model_Allen_10'
        # model_name = os.path.join(os.path.split(saveInfo['Dir'])[0], 'caiman-CNN', 'cnn_model'+cnt)
        print(model_name)

        # cm.stop_server(dview=dview)
        # c, dview, n_processes =\
        #     cm.cluster.setup_cluster(backend='local', n_processes=20,
        #                         single_thread=False)
        start_CNN = time()
        cnm3.estimates.evaluate_components(images, cnm3.params, dview=dview,model_name=model_name)
        #% update object with selected components
        cnm3.estimates.select_components(use_object=True)
        time_CNN = time()-start_CNN
        tottime = time_CNMF + time_CNN 
        # performance = Performance_metrics(GTMask[ID],cnm2.estimates.A,dims)
        # f1 = performance['f1_score']  

        #% save masks to mat file
        saveDir_CV = saveInfo['Dir'] + 'CV'+str(CV)+'\\'
        if not os.path.exists(saveDir_CV):
            os.makedirs(saveDir_CV)
        if save_results:
            sio.savemat(saveDir_CV+saveInfo['Name'],mdict={'Ab': (cnm3.estimates.A).toarray(),'ProcessTime':tottime, 'time_CNN':time_CNN},\
                oned_as='column', do_compression=True)

    # return f1


def Performance_metrics(GTMask,Ab,dims):
    _, _,_, _, performance,_ = register_ROIs(GTMask, Ab, dims, template1=None, template2=None, align_flag=False, 
                  D=None, max_thr = 0.2, use_opt_flow = False, thresh_cost=.5, # .7, #  
                  max_dist=30, enclosed_thr=None, print_assignment=False, 
                  plot_results=False, Cn=None, cmap='viridis')
    return performance


