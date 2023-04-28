import sys
import numpy as np
import time
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import keras
from keras import backend as K
# from sklearn.model_selection import train_test_split

# from CNN_classifier import CNN_classifier

import json as simplejson
from tensorflow.keras.models import model_from_json

import scipy.io as sio
import h5py

import tensorflow as tf
tf_version = int(tf.__version__[0])
if tf_version == 1:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config = config)
else: # tf_version == 2:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":
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
    list_Exp_ID = [ 'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
                    'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']
    nvideo = len(list_Exp_ID)
    # dir_parent = '../data/data_TENASPIS/original_masks'
    dir_parent = '../data/data_TENASPIS/added_refined_masks'
    dir_init = sys.argv[1] # 'SUNS_TUnCaT_SF25\\4816[1]th5\\output_masks'

    opt_th_cnn = False
    classifier = 'classifier_res0'
    num_frame = 0
    mask_option = 'Xmask'
    random_shuffle_channel = False
    num_mask_channel = 1 
    lam = 15
    drop_rate = '0.8exp(-{})'.format(lam) # sys.argv[1] # 
    if num_frame <= 1:
        img_option = 'avg'
    else:
        img_option = 'multi_frame'
    dir_CNN = os.path.join(dir_parent, 'GT Masks dropout {}/add_new_blockwise'.format(drop_rate))

    # model = CNN_classifier(input_shape, num_classes)
    list_dir_test = [os.path.join(dir_parent, dir_init, 'add_new_blockwise')]
    for dir_test in list_dir_test:
        dir_SUNS_out = os.path.join(dir_test, 'trained dropout {}'.format(drop_rate))
        if not os.path.exists(dir_SUNS_out):
            os.makedirs(dir_SUNS_out) 

        dir_save = dir_SUNS_out
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
                raise(ValueError('"img_option" must be in {}'.format({'avg','multi_frame'})))

            if mask_option == 'nomask':
                pass
            elif mask_option == 'mask':
                x_test = np.concatenate([x_test, mask_test], axis=0)
            elif mask_option == 'bmask':
                x_test = np.concatenate([x_test, mask_test], axis=0)
            elif mask_option == 'Xmask':
                x_test = np.concatenate([x_test, mask_test*image_test], axis=0)
            else:
                raise(ValueError('"mask_option" must be in {}'.format({'nomask','mask','Xmask','bmask'})))

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
            if len(y_test):
                pred_valid = (y_test[:,1] > th_cnn).squeeze()
            else:
                pred_valid = np.array([], dtype='bool')
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
