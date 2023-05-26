import numpy as np
import cv2

# @profile
def threshold_frame(frame, thred_inten, target_area=None, unmasked=None):
    ''' Threshold an image to a binary mask with a single connected region, 
        so that the area was equal to the target_area. 
        If simple thresholding gave multiple disjoint regions, 
        we iteratively lowered the threshold until the area of the region having 
        the most pixels in the outside region exceeded the average area of the initial masks. 
        After stopping this process, we retained that biggest disjoint region 
        at the final threshold as the binary mask of this segment. 
    Inputs: 
        frame (2D numpy.ndarray): the image to be binarized.
        thred_inten (float): The binarization threshold (to start with).
        target_area (int): The target area of the threshold mask.
        unmasked (2D numpy.ndarray): the image to be binarized.

    Outputs:
        traces (numpy.ndarray of float, shape = (T,n)): The raw traces of all neurons.
            Each column represents a trace of a neuron.
    '''

    Lym, Lxm = frame.shape
    if unmasked is None:
        unmasked = np.ones((Lym, Lxm),'bool')

    final_thred = thred_inten
    frame_thred = frame > thred_inten
    noisy = False
    nL, L, stats, _= cv2.connectedComponentsWithStats(frame_thred.astype('uint8'), connectivity=4)
    # L = L - 1

    if nL >= 1:
        # Find the thresholded region with the largest area
        list_area_L = stats[1:,4]
        max_area = list_area_L.max()
        if max_area < frame_thred.sum() / 2:
            noisy = True

        L_unmasked = (L * unmasked)
        L -= 1
        nL_masked, L_masked, stats_masked, _= cv2.connectedComponentsWithStats(L_unmasked.astype('uint8'), connectivity=4)
        if nL_masked == 1:
            frame_thred = L_unmasked.astype('bool')
        else:
            list_area_masked_L = stats_masked[1:,4]
            iLm = np.argmax(list_area_masked_L)
            max_area_masked = list_area_masked_L[iLm]
            core_region_masked = (L_masked == iLm + 1)
            candidates = (list_area_L >= max_area_masked).nonzero()[0]
            n_candidates = candidates.size
            if n_candidates == 0:
                core_region = core_region_masked
            elif n_candidates == 1:
                core_region = (L == candidates)
            else:
                masks_large = np.zeros([n_candidates, Lym, Lxm], 'bool')
                list_area_masked_large = np.zeros(n_candidates, 'uint32')
                for (ii, ni) in enumerate(candidates):
                    masks_large[ii, :, :] = (L == ni)
                    list_area_masked_large[ii] = (masks_large[ii, :, :] * unmasked).sum()
                ii = np.argmax(list_area_masked_large)
                core_region = masks_large[ii, :, :]

            # If the area of the core region is too small, iteratively lower the threshold
            if (target_area is None) or (max_area >= target_area): # or (len(target_area) == 0)
                frame_thred = core_region
            else:
                frame_thred_old = frame_thred
                while max_area < target_area:
                    final_thred = thred_inten
                    frame_thred_old = frame_thred
                    if thred_inten > 1:
                        thred_inten = thred_inten * 0.9
                    else:
                        thred_inten = thred_inten - 0.1
                    frame_thred = frame > thred_inten
                    nL, L, stats, _= cv2.connectedComponentsWithStats(frame_thred.astype('uint8'), connectivity=4)
                    L = L - 1
                    list_area_L = stats[1:,4]
                    max_area = list_area_L.max()

                nL, L, stats, _= cv2.connectedComponentsWithStats(frame_thred_old.astype('uint8'), connectivity=4)
                L = L - 1
                list_area_L_core = np.zeros(nL, 'uint32')
                yy, xx = core_region.nonzero()
                for (iy, ix) in zip(yy,xx):
                    val = L[iy, ix]
                    if val > -1:
                        list_area_L_core[val] += 1
                iL = np.argmax(list_area_L_core)
                frame_thred = (L == iL)

    # Fill the holes in the thresholded core region
    nL0, L0, stats, _= cv2.connectedComponentsWithStats(np.logical_not(frame_thred).astype('uint8'), connectivity=4)
    L0 = L0 - 1
    if nL0 > 1:
        list_area_L0 = stats[1:,4]
        iL0 = np.argmax(list_area_L0)
        frame_thred = (L0 != iL0)

    return frame_thred, noisy, final_thred
