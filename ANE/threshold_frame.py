import numpy as np
from scipy.ndimage import label
# import cv2


def threshold_frame(frame, thred_inten, target_area=None, unmasked=None):
    Lym, Lxm = frame.shape
    if unmasked is None:
        unmasked = np.ones((Lym, Lxm),'bool')

    final_thred = thred_inten
    frame_thred = frame > thred_inten
    noisy = False
    L, nL = label(frame_thred)
    # L, nL= cv2.connectedComponents(frame_thred, connectivity=4)

    if nL >= 1:
        # Find the thresholded region with the largest area
        list_area_L = np.zeros(nL)
        list_area_masked_L = np.zeros(nL)
        for iy in range(Lym):
            for ix in range(Lxm):
                val = L[iy, ix]
                if val:
                    list_area_L[val] = list_area_L[val] + 1
                    list_area_masked_L[val] = list_area_masked_L[val] + unmasked[iy, ix]
        max_area = list_area_L.max()
        if max_area < frame_thred.sum() / 2:
            noisy = True
        iLm = np.argmax(list_area_masked_L)
        core_region = (L == iLm)

        # If the area of the core region is too small, iteratively lower the threshold
        if (target_area is None) or (len(target_area) == 0) or (max_area >= target_area):
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
                L, nL = label(frame_thred)
                # L, nL= cv2.connectedComponents(frame_thred, connectivity=4)
                list_area_L = np.zeros(nL)
                for ix in range(Lxm):
                    for iy in range(Lym):
                        val = L[iy, ix]
                        if val:
                            list_area_L[val] = list_area_L[val] + 1
                max_area = list_area_L.max()

            L, nL = label(frame_thred_old)
            # L, nL= cv2.connectedComponents(frame_thred_old, connectivity=4)
            list_area_L_core = np.zeros(nL)
            yy, xx = core_region.nonzero()
            # for ii in range(len(xx)):
            #     val = L[yy[ii], xx[ii]]
            for (iy, ix) in np.zip(yy,xx):
                val = L[iy, ix]
                if val:
                    list_area_L_core[val] = list_area_L_core[val] + 1
            iL = np.argmax(list_area_L_core)
            frame_thred = (L == iL)

    # Fill the holes in the thresholded core region
    L0, nL0 = label(np.logical_not(frame_thred))
    # L0, nL0= cv2.connectedComponents(np.logical_not(frame_thred), connectivity=4)
    if nL0 > 1:
        list_area_L0 = np.zeros(nL0)
        for iy in range(Lym):
            for ix in range(Lxm):
                val = L0[iy, ix]
                if val:
                    list_area_L0[val] = list_area_L0[val] + 1
        iL0 = np.argmax(list_area_L0)
        frame_thred = (L0 != iL0)

    return frame_thred, noisy, final_thred
