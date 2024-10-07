import numpy as np
import tifffile
import h5py
import os
import scipy.io as sio
import cv2

median_area_target = 377

if 1:
	dir_parent = r'D:\data_TENASPIS\added_refined_masks'
	list_Exp_ID = [ 'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
					'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']
	median_area = 228 # 305 # 

	dir_masks = os.path.join(dir_parent, 'GT Masks')
	dir_save = os.path.join(dir_parent, 'DeepWonder_scale_full')
	for Exp_ID in list_Exp_ID:
		dir_sub = os.path.join(dir_save, Exp_ID)
		if not os.path.exists(dir_sub):
			os.makedirs(dir_sub)
		dir_saved_masks = os.path.join(dir_sub, 'GT Masks')
		if not os.path.exists(dir_saved_masks):
			os.makedirs(dir_saved_masks)

		# Load masks
		filename_masks = os.path.join(dir_masks, 'FinalMasks_'+Exp_ID+'.mat')
		try:
			file_masks = sio.loadmat(filename_masks)
			FinalMasks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
		except:
			file_masks = h5py.File(filename_masks, 'r')
			FinalMasks = np.array(file_masks['FinalMasks']).astype('bool')
			file_masks.close()
		
		# median_area = np.median(FinalMasks.sum(2).sum(1))
		mag = np.sqrt(median_area_target/median_area)
		(N, Ly, Lx) = FinalMasks.shape
		Lys = int(Ly * mag / 2) * 2
		Lxs = int(Lx * mag / 2) * 2

		# Resize the masks
		FinalMasks_resize = np.zeros((N, Lys, Lxs), 'bool')
		for n in range(N):
			FinalMasks_resize[n] = cv2.resize(FinalMasks[n].astype('uint8'), (Lxs, Lys))
		sio.savemat(os.path.join(dir_saved_masks, 'FinalMasks_'+Exp_ID+'.mat'),\
			{'FinalMasks':FinalMasks_resize.transpose([2,1,0])}, do_compression=True)

		# open the h5 file in read-only mode
		with h5py.File(os.path.join(dir_parent,Exp_ID+'.h5'), 'r') as f:
			# get list of dataset names in file
			# print(list(f.keys()))
			dataset = f['mov']
			data = dataset[:]

			# Resize the masks
			T = data.shape[0]
			data_resize = np.zeros((T, Lys, Lxs), data.dtype)
			for t in range(T):
				data_resize[t] = cv2.resize(data[t], (Lxs, Lys))

			# Convert to uint16
			data_resize = data_resize / data_resize.max() * (2**16 - 1)
			data_resize = data_resize.astype('uint16')

		# Save the array as a TIFF stack
		tifffile.imwrite(os.path.join(dir_sub, Exp_ID+'.tiff'), data_resize, imagej=True)
		print((T, Lys, Lxs, N))