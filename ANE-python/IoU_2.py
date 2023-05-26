import numpy as np
from scipy import sparse


def IoU_2(GTMasks, Masks):
    '''Calculate the recall, precision, and F1 score of segmented neurons by comparing with ground truth.

    Inputs: 
        GTMasks (sparse.csr_matrix): Ground truth masks.
        Masks (sparse.csr_matrix): Segmented masks.

    Outputs:
        IoU (np.array): IoU between each pair of neurons. 
    '''
    if 'bool' in str(Masks.dtype): # bool cannot be used to calculate IoU
        Masks = Masks.astype('uint32')
    if 'bool' in str(GTMasks.dtype):
        GTMasks = GTMasks.astype('uint32')
    NGT = GTMasks.shape[0] # Number of GT neurons
    NMask = Masks.shape[0] # Number of segmented neurons
    a1 = np.repeat(GTMasks.sum(axis=1).A, NMask, axis=1)
    a2 = np.repeat(Masks.sum(axis=1).A.T, NGT, axis=0)
    intersectMat = GTMasks.dot(Masks.transpose()).A
    unionMat = a1 + a2 - intersectMat
    IoU = intersectMat/unionMat # IoU between each pair of neurons

    return IoU
