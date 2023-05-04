import numpy as np


def frame_weight_blockwise_mm(fn_video, video_dtype, video_shape, fn_masks, masks_shape, d, leng):
    T, Ly, Lx = video_shape
    # N, Ly, Lx = masks_shape
    fp_video = np.memmap(fn_video, dtype=video_dtype, mode='r', shape=video_shape)
    fp_masks = np.memmap(fn_masks, dtype='bool', mode='r', shape=masks_shape)
    npatchx = np.ceil(Lx / leng) - 1
    npatchy = np.ceil(Ly / leng) - 1

    # Calculate the weight of each frame
    list_weight = np.zeros((npatchy, npatchx), 'object')
    list_weight_trace = np.zeros((npatchy, npatchx), 'object')
    list_d_diff = np.zeros((npatchy, npatchx), 'object')
    list_weight_frame = np.zeros((npatchy, npatchx), 'object')

    for iy in range(npatchy):
        for ix in range(npatchx):
            # Calculate the weight from maximum intensity of each frame
            xmin = np.minimum(Lx - 2 * leng + 1, (ix - 1) * leng + 1) - 1
            xmax = np.minimum(Lx, (ix + 1) * leng) - 1
            ymin = np.minimum(Ly - 2 * leng + 1, (iy - 1) * leng + 1) - 1
            ymax = np.minimum(Ly, (iy + 1) * leng) - 1
            masks_sub = fp_masks[:, ymin:ymax+1, xmin:xmax+1]
            neighbors = masks_sub.sum(2).sum(1) > 0
            masks_neighbors = masks_sub[neighbors, :, :]
            _, Lym, Lxm = masks_neighbors.shape
            union_neighbors = masks_neighbors.sum(0) > 0
            union_neighbors_2 = union_neighbors.ravel()
            video_sub = fp_video[:, ymin:ymax+1, xmin:xmax+1]
            video_sub_2 = video_sub.reshape((T, Lym * Lxm))
            nearby_outside_2 = np.logical_not(union_neighbors_2)

            order_compare = np.arange(2, 10)
            if np.amin(union_neighbors_2.sum(), nearby_outside_2.sum()) < order_compare.max():
                list_weight_frame[iy, ix] = 0
            else:
                sort_inside = np.flip(np.sort(video_sub_2[:, union_neighbors_2], 1), 1)
                max_inside = sort_inside[order_compare, :]
                sort_outside = np.flip(np.sort(video_sub_2[:, nearby_outside_2], 1), 1)
                max_outside = sort_outside[:, order_compare]
                list_weight_frame[iy, ix] = np.maximum(0, np.minimum(max_outside - max_inside))

            # Calculate the weight from trace
            d_out = video_sub_2[:, nearby_outside_2].mean(1)
            d_neurons_max = np.maximum(d[:, neighbors], 0)
            d_diff = d_out - d_neurons_max
            list_d_diff[iy, ix] = d_diff
            list_weight_trace[iy, ix] = np.maximum(0, d_diff)
            # Combined weight
            list_weight[iy, ix] = np.maximum(list_weight_trace[iy, ix], list_weight_frame[iy, ix])

    return list_weight, list_weight_trace, list_weight_frame
