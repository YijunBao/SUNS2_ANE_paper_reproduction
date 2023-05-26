import os
import numpy as np
import multiprocessing as mp


def mean_trace(nn, fn_video, video_dtype, video_shape, fn_masks, mask_shape):
    ''' Calculate the traces of the neuron, background, and outside region for a neuron. 
    Inputs: 
        nn (int): The index of the neuron of interest.
        fn_video (str): Memory mapping file name of the input video.
        video_dtype (str): The data type of the input video.
        video_shape (tuple of int, shape = (3,)): The shape of the input video.
        fn_masks (str): Memory mapping file name of the binary spatial masks of all neurons.
        mask_shape (tuple of int, shape = (3,)): The shape of the binary spatial masks.

    Outputs:
        trace (numpy.ndarray of float, shape = (T,)): The raw trace of the neuron.
    '''

    (T, Lx, Ly) = video_shape
    n = mask_shape[0]
    # Reconstruct the video, masks, and corrdinates from memory mapping files.
    fp_video = np.memmap(fn_video, dtype=video_dtype, mode='r', shape=(T,Lx*Ly))
    fp_masks = np.memmap(fn_masks, dtype='bool', mode='r', shape=(n,Lx*Ly))
    mask = fp_masks[nn]

    trace = fp_video[:, mask].mean(1) # Neuron trace
    return trace


def traces_from_masks_mmap(fn_video, video_dtype, video_shape, fn_masks, masks_shape, useMP, p):
    ''' Calculate the traces of the neuron, background, and outside region for all neurons. 
    Inputs: 
        fn_video (str): Memory mapping file name of the input video.
        video_dtype (str): The data type of the input video.
        video_shape (tuple of int, shape = (3,)): The shape of the input video.
        fn_masks (str): Memory mapping file name of the binary spatial masks of all neurons.
        masks_shape (tuple of int, shape = (3,)): The shape of the neuron masks.
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        p (multiprocessing.Pool, default to None): 

    Outputs:
        traces (numpy.ndarray of float, shape = (T,n)): The raw traces of all neurons.
            Each column represents a trace of a neuron.
    '''

    ncells = masks_shape[0]

    # Calculate the traces of the neuron, background, and outside region for each neuron. 
    if useMP:
        if not p: # start a multiprocessing.Pool
            p = mp.Pool(mp.cpu_count())
            closep = True
        else:
            closep = False
        results = p.starmap(mean_trace, [(nn, fn_video, video_dtype, video_shape, \
            fn_masks, masks_shape) for nn in range(ncells)], chunksize=1)
        # p.close()
    else:
        results = []
        for nn in range(ncells):
            results.append(mean_trace(nn, fn_video, video_dtype, video_shape, \
                fn_masks, masks_shape))

    traces = np.stack(results, 1)
    if useMP and closep:
        p.close()
        p.join()
    return traces
