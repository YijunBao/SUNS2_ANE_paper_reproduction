import sys
import numpy as np
from threshold_frame import threshold_frame
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy import sparse
import multiprocessing as mp

sys.path.insert(1, '../SUNS2') # the path containing "suns" folder
from suns.PostProcessing.combine import piece_neurons_IOU
    
# @profile
def find_missing_blockwise_mm(fn_video, video_dtype, video_shape, fn_masks, masks_shape,\
    xmin,xmax,ymin,ymax,weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split): 

    fp_video = np.memmap(fn_video, dtype=video_dtype, mode='r', shape=video_shape)
    fp_masks = np.memmap(fn_masks, dtype='bool', mode='r', shape=masks_shape)
    T,Ly,Lx = video_shape
    # Ly,Lx,N = masks.shape
    # xrange = np.arange(xmin,xmax+1)
    # yrange = np.arange(ymin,ymax+1)
    video_sub = fp_video[:, ymin:ymax+1, xmin:xmax+1]
    masks_sub = fp_masks[:, ymin:ymax+1, xmin:xmax+1]
    neighbors = masks_sub.sum(2).sum(1).squeeze() > 0
    masks_neighbors = masks_sub[neighbors,:,:]
    _,Lym,Lxm = masks_neighbors.shape
    unmasked = np.logical_not(masks_neighbors.sum(0))
    q = 1 - avg_area / (Lym * Lxm)
    near_zone = np.zeros((Lym,Lxm), 'bool')
    near_zone[int(np.round(Lym / 4 + 1)-1):int(np.round(Lym * 3 / 4)), int(np.round(Lxm / 4 + 1)-1):int(np.round(Lxm * 3 / 4))] = True
    far_zone = np.logical_not(near_zone)
    near_zone_2 = near_zone.ravel()
    far_zone_2 = far_zone.ravel()
    select_frames = (weight > 0).nonzero()[0]
    select_frames_order = np.flip(np.argsort(weight[select_frames]))
    select_frames_sort = select_frames[select_frames_order]

    # Exclude frames that do not have qualified binary masks
    list_good = np.zeros(num_avg, 'uint32')
    g = 0
    for t in range(len(select_frames_sort)):
        t_frame = select_frames_sort[t]
        frame = video_sub[t_frame,:,:]
        thred_inten = np.quantile(frame,q)
        frame_thred,noisy,_ = threshold_frame(frame,thred_inten,avg_area,unmasked)
        if not noisy:
            nearby = np.logical_and(frame_thred,far_zone).sum() < np.logical_and(frame_thred,near_zone).sum() * 2
            if nearby:
                list_good[g] = t_frame
                g = g + 1
                if g >= num_avg - 1:
                    break
    
    if g >= num_avg:
        select_frames_sort = list_good
    else:
        select_frames_sort = list_good[0:g]
    
    ## Update tiles
    n_select = len(select_frames_sort)
    video_sub_sort = video_sub[select_frames_sort,:,:]
    mask_update_select = np.zeros((n_select,Lym,Lxm), 'bool')
    list_noisy = np.zeros(n_select, 'bool')
    for kk in range(n_select):
        frame = video_sub_sort[kk,:,:]
        thred_inten = np.quantile(frame,q)
        frame_thred,noisy,_ = threshold_frame(frame,thred_inten,avg_area,unmasked)
        mask_update_select[kk,:,:] = frame_thred
        list_noisy[kk] = noisy
    
    ## Clustering
    select_frames_sort_full = select_frames_sort
    if n_select == 0:
        classes = []
        list_far = np.zeros(0, 'bool')
        list_class_frames = np.array([])
    else:
        if n_select == 1:
            classes = 0
            list_far = np.zeros(0, 'bool')
            list_class_frames = np.array([])
        else:
            mask_update_select_2 = mask_update_select.reshape((n_select, Lym * Lxm))
            mask_update_select_2[list_noisy,:] = 0
            dist = pdist(mask_update_select_2.astype('float64'),metric='jaccard')
            tree = linkage(dist,method='average')
            classes = fcluster(tree,thj,criterion='distance') - 1
            dist_2 = squareform(dist)
            n_class = classes.max() + 1
            valid_cluster = np.zeros(n_class, 'bool')
            list_classes = np.array([(classes == x).nonzero()[0] for x in range(n_class)])
            list_avg_mask_2 = np.zeros(n_class, 'object')
            list_far = np.zeros(n_select, 'bool')
            for c in range(n_class):
                class_frames = list_classes[c]
                dist_2_c = dist_2[np.ix_(class_frames,class_frames)] + np.eye(len(class_frames))
                min_dist_2_c = np.amin(dist_2_c)
                if min_dist_2_c < thj_inclass:
                    valid_cluster[c] = True
                    avg_mask_2 = mask_update_select_2[class_frames,:].sum(0)
                    avg_mask_2 = avg_mask_2 > 0.5 * avg_mask_2.max()
                    list_avg_mask_2[c] = avg_mask_2
                    # Remove potential missing neurons that are much closer to other neurons
                    nearby = np.logical_and(avg_mask_2,far_zone_2).sum() < np.logical_and(avg_mask_2,near_zone_2).sum() * 2
                    if not nearby :
                        valid_cluster[c] = False
                        list_far[class_frames] = True
                    
            if valid_cluster.any():
                avg_mask_2_all = np.stack(list_avg_mask_2[valid_cluster], 0)
                list_classes_valid = list_classes[valid_cluster]
                _,list_class_frames = piece_neurons_IOU(sparse.csr_matrix(avg_mask_2_all),0.5,th_IoU_split,list_classes_valid)
            else:
                list_class_frames = np.array([])
    
    ## Multiple neurons
    n_class = len(list_class_frames)
    classes = np.zeros(n_select, 'uint32')
    select_frames_sort = np.zeros(n_class, 'object')
    for c in range(n_class):
        classes[list_class_frames[c]] = c
        select_frames_sort[c] = select_frames_sort_full[list_class_frames[c]]
    
    classes[list_far] = - 1
    avg_frame = np.zeros(n_class, 'object')
    mask_update = np.zeros(n_class, 'object')
    select_frames_class = np.zeros(n_class, 'object')
    select_weight_calss = np.zeros(n_class, 'object')
    for c in range(n_class):
        select_frames_class[c] = video_sub[select_frames_sort[c],:,:]
        select_weight_calss[c] = weight[select_frames_sort[c]]
        avg_frame[c] = sum([x*y for x,y in zip(select_frames_class[c],select_weight_calss[c])]) / select_weight_calss[c].sum()
        avg_frame_use = avg_frame[c]
        thred_inten = np.quantile(avg_frame_use,q)
        mask_update[c],_,_ = threshold_frame(avg_frame_use,thred_inten,avg_area,unmasked)
    
    ## confirm select multi
    mask_new_full = np.zeros((n_class,Ly,Lx), 'bool')
    mask_new_crop = np.zeros((n_class,Lym,Lxm), 'bool')
    image_new_crop = np.zeros((n_class,Lym,Lxm),'float32')
    for c in range(n_class):
        mask_new_full[c,ymin:ymax+1, xmin:xmax+1] = mask_update[c]
        mask_new_crop[c,:,:] = mask_update[c]
        image_new_crop[c,:,:] = avg_frame[c]
    
    return image_new_crop,mask_new_crop,mask_new_full,select_frames_class,select_weight_calss


def use_find_missing_blockwise_mm(fn_video, video_dtype, video_shape, fn_masks, masks_shape,\
    ii, npatchx, npatchy, leng, list_weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split): 

    _, Ly, Lx = video_shape
    iy = ii // npatchx
    ix = ii % npatchx
    xmin = np.minimum(Lx - 2 * leng + 1, (ix) * leng + 1)-1
    xmax = np.minimum(Lx, (ix + 2) * leng)-1
    ymin = np.minimum(Ly - 2 * leng + 1, (iy) * leng + 1)-1
    ymax = np.minimum(Ly, (iy + 2) * leng)-1
    weight = list_weight[iy, ix]
    image_new_crop, mask_new_crop, mask_new_full, select_frames_class, select_weight_calss = \
        find_missing_blockwise_mm(fn_video, video_dtype, video_shape, fn_masks, masks_shape,\
            xmin, xmax, ymin, ymax, weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split)
    n_class = image_new_crop.shape[0]
    locations = np.repeat([[xmin, xmax, ymin, ymax]], n_class, 0)
    return image_new_crop,mask_new_crop,mask_new_full,select_frames_class,select_weight_calss, locations


def find_missing_blockwise(fn_video, video_dtype, video_shape, fn_masks, masks_shape,\
    leng, list_weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split, useMP, p): 

    _, Ly, Lx = video_shape
    npatchx = int(np.ceil(Lx / leng) - 1)
    npatchy = int(np.ceil(Ly / leng) - 1)
    npatch = npatchx * npatchy
    patch_size = (npatchy, npatchx)

    if useMP:
        if not p: # start a multiprocessing.Pool
            p = mp.Pool(mp.cpu_count())
            closep = True
        else:
            closep = False
        segs = p.starmap(use_find_missing_blockwise_mm, [(fn_video, video_dtype, video_shape, fn_masks, masks_shape,\
            ii, npatchx, npatchy, leng, list_weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split) \
            for ii in range(npatch)], chunksize=1)
        # list_added_images_crop = np.array([x[0] for x in segs], 'object')#.reshape(patch_size)
        # list_added_crop = np.array([x[1] for x in segs], 'object')#.reshape(patch_size)
        # list_added_full = np.array([x[2] for x in segs], 'object')#.reshape(patch_size)
        # list_added_frames = np.array([x[3] for x in segs], 'object')#.reshape(patch_size)
        # list_added_weights = np.array([x[4] for x in segs], 'object')#.reshape(patch_size)
        # list_locations = np.array([x[5] for x in segs], 'object')#.reshape(patch_size)
        list_added_images_crop = np.zeros(npatch, 'object')
        list_added_crop = np.zeros(npatch, 'object')
        list_added_full = np.zeros(npatch, 'object')
        list_added_frames = np.zeros(npatch, 'object')
        list_added_weights = np.zeros(npatch, 'object')
        list_locations = np.zeros(npatch, 'object')
        for (ii, x) in enumerate(segs):
            list_added_images_crop[ii] = x[0]
            list_added_crop[ii] = x[1]
            list_added_full[ii] = x[2]
            list_added_frames[ii] = x[3]
            list_added_weights[ii] = x[4]
            list_locations[ii] = x[5]
        if closep:
            p.close()
            p.join()
    
    else:
        list_added_images_crop = np.zeros(patch_size, 'object')
        list_added_crop = np.zeros(patch_size, 'object')
        list_added_full = np.zeros(patch_size, 'object')
        list_added_frames = np.zeros(patch_size, 'object')
        list_added_weights = np.zeros(patch_size, 'object')
        list_locations = np.zeros(patch_size, 'object')
        ##
        for iy in range(npatchy):
            for ix in range(npatchx):
                xmin = np.minimum(Lx - 2 * leng + 1, (ix) * leng + 1)-1
                xmax = np.minimum(Lx, (ix + 2) * leng)-1
                ymin = np.minimum(Ly - 2 * leng + 1, (iy) * leng + 1)-1
                ymax = np.minimum(Ly, (iy + 2) * leng)-1
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
    masks_added_full = np.concatenate(list_added_full.ravel(), 0)
    images_added_crop = np.concatenate(list_added_images_crop.ravel(), 0)
    masks_added_crop = np.concatenate(list_added_crop.ravel(), 0)
    patch_locations = np.concatenate(list_locations.ravel(), 0)
    added_frames = np.concatenate(list_added_frames.ravel())
    added_weights = np.concatenate(list_added_weights.ravel())

    return masks_added_full, images_added_crop, masks_added_crop, patch_locations, added_frames, added_weights
