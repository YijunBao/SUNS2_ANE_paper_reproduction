import sys
import numpy as np
import time
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import keras
from keras import backend as K
# from sklearn.model_selection import train_test_split

import tensorflow as tf
# from CNN_classifier import CNN_classifier

import json as simplejson
from keras.models import model_from_json

import scipy.io as sio
import h5py

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config = config)

if __name__ == "__main__":
    list_img_option = {'avg','multi_frame'}
    list_mask_option = {'nomask','mask','Xmask','bmask'}
    #%
    batch_size = 16
    # num_classes = 2
    # epochs = 2000
    # test_fraction = 0.25
    # augmentation = True
    # th_cnn = 0.5
    # input image dimensions
    # Lx, Ly = 50, 50

    # %%the data, shuffled and split between train and test sets
    # list_radius = [8,10,8,6] # 
    # list_rate_hz = [10,15,7.5,5] # 
    # list_decay_time = [0.4, 0.5, 0.4, 0.75]
    # Dimens = [(120,120),(80,80), (88,88),(192,240)]
    # list_nframes = [6000, 9000, 9000, 1500]

    list_Exp_ID = [ 'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
                    'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']
    nvideo = len(list_Exp_ID)
    list_opt_th_cnn = [True, False]
    for opt_th_cnn in list_opt_th_cnn:
        for version in [0]: # range(2,9): # range(0,10): # 
            # num_frame = version
            # folder = 'classifier_res{}'.format(version)
            # folder = '{}_{}frames'.format('classifier_res0', version)
            classifier = sys.argv[1] # 'classifier_res1' #  
            num_frame = int(sys.argv[2]) # 1 # 
            mask_option = sys.argv[3] # 'mask_thb' # 
            random_shuffle_channel = bool(int(sys.argv[4])) # True
            num_mask_channel = int(sys.argv[5]) # 4 # 
            folder_sub = sys.argv[6] # '_unmask' # 
            folder = '{}_{}+{} frames'.format(classifier, num_frame, num_mask_channel)
            if random_shuffle_channel:
                folder = folder + '_shuffle'
            if num_frame <= 1:
                img_option = 'avg'
            else:
                img_option = 'multi_frame'
            sub = img_option + '_' + mask_option
            # dir_CNN = os.path.join('D:\\data_TENASPIS\\original_masks\\GT Masks\\add_new_blockwise{}\\classifiers_{}'.format(folder_sub, sub), folder)
            dir_CNN = os.path.join('D:\\data_TENASPIS\\added_refined_masks\\GT Masks\\add_new_blockwise{}\\classifiers_{}'.format(folder_sub, sub), folder)

            # model = CNN_classifier(input_shape, num_classes)
            for th_SNR in [4]: # range(3,6):
                list_dir_test = []
                # list_dir_test.append('D:\\data_TENASPIS\\original_masks\\complete_TUnCaT_SF25\\4816[1]th{}\\output_masks\\add_new_blockwise{}'.format(th_SNR, folder_sub))
                list_dir_test.append('D:\\data_TENASPIS\\added_refined_masks\\complete_TUnCaT_SF25\\4816[1]th{}\\output_masks\\add_new_blockwise{}'.format(th_SNR, folder_sub))
                for dir_test in list_dir_test:
                    if opt_th_cnn:
                        dir_save = os.path.join(dir_test, sub, folder) # 
                    else:
                        dir_save = os.path.join(dir_test, sub+'_0.5', folder) # 

                    if not os.path.exists(dir_save):
                        os.makedirs(dir_save) 

                    for (cv,Exp_ID) in enumerate(list_Exp_ID): #
                        training_output = sio.loadmat(os.path.join(dir_CNN, "training_output_cv{}.mat".format(cv)))
                        if opt_th_cnn:
                            th_cnn = training_output['th_cnn_best']
                        else:
                            th_cnn = 0.5
                        model_name = os.path.join(dir_CNN, 'cnn_model_cv' + str(cv))
                        model_file = model_name + ".json"
                        with open(model_file, 'r') as json_file:
                            print('USING MODEL:' + model_file)
                            loaded_model_json = json_file.read()
                        loaded_model = model_from_json(loaded_model_json)
                        loaded_model.load_weights(model_name + '.h5')

                        added_auto = sio.loadmat(os.path.join(dir_test, Exp_ID+'_added_auto_blockwise.mat'))
                        image_test = np.expand_dims(np.array(added_auto['images_added_crop']), 0)
                        added_frames = added_auto['added_frames'][0]
                        mask_added_test = np.array(added_auto['masks_added_crop'])
                        # mask_center_test = np.array(added_auto['masks_center_crop'])
                        if num_mask_channel == 1:
                            mask_test = np.expand_dims(mask_added_test.astype(image_test.dtype), 0)
                        elif num_mask_channel == 2:
                            mask_no_neighbors_test = np.logical_not(np.array(added_auto['masks_neighbors_crop']))
                            mask_test = np.stack((mask_added_test, mask_no_neighbors_test), axis=0).astype(image_test.dtype)

                        start = time.time()
                        if img_option == 'multi_frame':
                            (_, Ly, Lx, n) = mask_test.shape
                            x_test = np.zeros((num_frame, Ly, Lx, n), dtype = image_test.dtype)
                            for i in range(n):
                                temp_frames = added_frames[i].transpose((2,0,1))
                                num_useful = temp_frames.shape[0]
                                if num_useful < num_frame:
                                    x_test[:num_useful, :, :, i] = temp_frames
                                else:
                                    x_test[:, :, :, i] = temp_frames[:num_frame]
                        elif img_option == 'avg':
                            x_test = image_test
                        else:
                            raise(ValueError('"img_option" must be in {}'.format(list_img_option)))

                        if mask_option == 'nomask':
                            pass
                        elif mask_option == 'mask':
                            x_test = np.concatenate([x_test, mask_test], axis=0)
                        elif mask_option == 'bmask':
                            x_test = np.concatenate([x_test, mask_test], axis=0)
                        elif mask_option == 'Xmask':
                            x_test = np.concatenate([x_test, mask_test*image_test], axis=0)
                        else:
                            raise(ValueError('"mask_option" must be in {}'.format(list_mask_option)))

                        if x_test.ndim == 3:
                            x_test = np.expand_dims(x_test, -1)
                        x_test = x_test.transpose((3,2,1,0))
                        (n, Lx, Ly, c) = x_test.shape

                        if K.image_data_format() == 'channels_first':
                            x_test = x_test.reshape(x_test.shape[0], c, Lx, Ly)
                            input_shape = (c, Lx, Ly)
                        else:
                            x_test = x_test.reshape(x_test.shape[0], Lx, Ly, c)
                            input_shape = (Lx, Ly, c)
                        
                        x_test = x_test.astype('float32')

                        # %%
                        y_test = loaded_model.predict(
                            x_test, batch_size=batch_size, verbose=1)
                        pred_valid = (y_test[:,1] > th_cnn).squeeze()
                        finish = time.time()
                        time_CNN = finish - start
                        print(pred_valid.sum(), 'added neurons out of', x_test.shape[0], 'candidates')

                        #%
                        # y_test = np.array(added_auto['list_valid']).squeeze()
                        # num_GT = sum(y_test)
                        # num_pred = sum(pred_valid)
                        # num_both = sum(np.logical_and(pred_valid,y_test))
                        # recall = num_both / num_GT
                        # precision  = num_both / num_pred
                        # f1 = 2/(1/recall + 1/precision)
                        # print(f1)
                    
                        sio.savemat(os.path.join(dir_save,"CNN_predict_{}_cv{}.mat".format(Exp_ID, cv)), 
                            {'pred_valid':pred_valid, 'time_CNN':time_CNN})
