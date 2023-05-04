import sys
import numpy as np
from threshold_frame import threshold_frame
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy import sparse

sys.path.insert(1, '../SUNS2') # the path containing "suns" folder
from suns.PostProcessing.combine import piece_neurons_IOU
    

def find_missing_blockwise_mm(fn_video, video_dtype, video_shape, fn_masks, masks_shape,\
    xmin,xmax,ymin,ymax,weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split): 
    fp_video = np.memmap(fn_video, dtype=video_dtype, mode='r', shape=video_shape)
    fp_masks = np.memmap(fn_masks, dtype='bool', mode='r', shape=masks_shape)
    T,Ly,Lx = video_shape
    # Ly,Lx,N = masks.shape
    xrange = np.arange(xmin,xmax+1)
    yrange = np.arange(ymin,ymax+1)
    video_sub = fp_video[:, yrange, xrange]
    masks_sub = fp_masks[:, yrange, xrange]
    neighbors = masks_sub.sum(2).sum(1).squeeze() > 0
    masks_neighbors = masks_sub[neighbors,:,:]
    _,Lym,Lxm = masks_neighbors.shape
    unmasked = np.logical_not(masks_neighbors.sum(2))
    q = 1 - avg_area / (Lym * Lxm)
    near_zone = np.zeros((Lym,Lxm), 'bool')
    near_zone[np.round(Lym / 4 + 1)-1:np.round(Lym * 3 / 4), np.round(Lxm / 4 + 1)-1:np.round(Lxm * 3 / 4)] = True
    far_zone = np.logical_not(near_zone)
    near_zone_2 = near_zone.ravel()
    far_zone_2 = far_zone.ravel()
    select_frames = (weight > 0).nonzero()
    select_frames_order = np.flip(np.argsort(weight[select_frames]))
    select_frames_sort = select_frames[select_frames_order]

    # Exclude frames that do not have qualified binary masks
    list_good = np.zeros(num_avg)
    g = 0
    for t in np.arange(len(select_frames_sort)):
        t_frame = select_frames_sort(t)
        frame = video_sub[t_frame,:,:]
        thred_inten = np.quantile(frame,q)
        frame_thred,noisy,_ = threshold_frame(frame,thred_inten,avg_area,unmasked)
        if not noisy:
            nearby = np.logical_and(frame_thred,far_zone).sum() < np.logical_and(frame_thred,near_zone).sum() * 2
            if nearby:
                g = g + 1
                list_good[g] = t_frame
                if g >= num_avg:
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
        list_far = np.zeros((1,0), 'bool')
        list_class_frames = np.array([])
    else:
        if n_select == 1:
            classes = 0
            list_far = np.zeros((1,0), 'bool')
            list_class_frames = np.array([])
        else:
            mask_update_select_2 = mask_update_select.reshape((n_select, Lym * Lxm))
            mask_update_select_2[:,list_noisy] = 0
            dist = pdist(mask_update_select_2.astype('float64'),metric='jaccard')
            tree = linkage(dist,method='average')
            classes = fcluster(tree,thj,criterion='distance')
            dist_2 = squareform(dist)
            n_class = classes.max()
            valid_cluster = np.zeros(n_class, 'bool')
            list_classes = [(classes == x).nonzero() for x in range(classes.max())]
            list_avg_mask_2 = np.zeros(n_class, 'object')
            list_far = np.zeros(n_select, 'bool')
            for c in np.arange(n_class):
                class_frames = list_classes[c]
                dist_2_c = dist_2[class_frames,class_frames] + np.eye(len(class_frames))
                min_dist_2_c = np.amin(dist_2_c)
                if min_dist_2_c < thj_inclass:
                    valid_cluster[c] = True
                    avg_mask_2 = mask_update_select_2[:,class_frames].sum(1).T
                    avg_mask_2 = avg_mask_2 > 0.5 * avg_mask_2.max()
                    list_avg_mask_2[c] = avg_mask_2
                    # Remove potential missing neurons that are much closer to other neurons
                    nearby = np.logical_and(avg_mask_2,far_zone_2).sum() < np.logical_and(avg_mask_2,near_zone_2).sum() * 2
                    if not nearby :
                        valid_cluster[c] = False
                        list_far[class_frames] = True
                    
            avg_mask_2_all = np.stack(list_avg_mask_2[valid_cluster], 0)
            list_classes_valid = list_classes[valid_cluster]
            _,list_class_frames = piece_neurons_IOU(sparse.csr_matrix(avg_mask_2_all),0.5,th_IoU_split,list_classes_valid)
    
    ## Multiple neurons
    n_class = len(list_class_frames)
    classes = np.zeros(n_select)
    select_frames_sort = np.zeros(n_class, 'object')
    for c in range(n_class):
        classes[list_class_frames[c]] = c
        select_frames_sort[c] = select_frames_sort_full(list_class_frames[c])
    
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
        mask_new_full[c,yrange,xrange] = mask_update[c]
        mask_new_crop[c,:,:] = mask_update[c]
        image_new_crop[c,:,:] = avg_frame[c]
    
    return image_new_crop,mask_new_crop,mask_new_full,select_frames_class,select_weight_calss