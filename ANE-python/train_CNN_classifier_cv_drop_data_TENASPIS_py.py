import sys
import numpy as np
import time
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras
from keras import backend as K
from sklearn.model_selection import train_test_split

import tensorflow as tf
from image_preprocessing_keras_3mask import ImageDataGenerator
# from CNN_classifier import CNN_classifier as CNN_classifier

import json as simplejson
# from keras.models import model_from_json
from sklearn.utils import class_weight as cw
from keras.utils.np_utils import to_categorical

import scipy.io as sio
import h5py

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
    # sys.argv = ['.py', 'classifier_res0', '0', 'Xmask', '0', '2', '_weighted_sum_unmask', '0.8exp(-3)']
    classifier = 'classifier_res0'
    num_frame = 0
    mask_option = 'Xmask'
    random_shuffle_channel = False
    num_mask_channel = 1 

    # start = time.time()
    if num_frame <= 1:
        img_option = 'avg'
        random_shuffle_channel = False
    else:
        img_option = 'multi_frame'
    exec('from CNN_classifier import CNN_' + classifier + ' as CNN_classifier')

    #%
    batch_size = 16
    num_classes = 2
    epochs = 2000
    # test_fraction = 0.25
    augmentation = True
    # input image dimensions
    # Lx, Ly = 50, 50

    # %%the data, shuffled and split between train and test sets
    list_Exp_ID = [ 'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
                    'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']
    lam = 15
    drop_rate = '0.8exp(-{})'.format(lam) # sys.argv[1] # 
    dir_parent = '../data/data_TENASPIS/added_refined_masks'
    # dir_parent = '../data/data_TENASPIS/original_masks'
    dir_video = os.path.join(dir_parent, 'GT Masks dropout {}/add_new_blockwise'.format(drop_rate))
    nvideo = len(list_Exp_ID) # number of videos used for cross validation

    list_th_cnn = np.arange(0.2, 0.81, 0.05)
    save_dir = dir_video
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    list_x_train = []
    list_y_train = []
    for cv in range(nvideo):
        Exp_ID = list_Exp_ID[cv]
        added_auto = sio.loadmat(os.path.join(dir_video, Exp_ID+'_added_CNNtrain_blockwise.mat'))
        y_train = np.array(added_auto['list_valid'])
        image_train = np.expand_dims(np.array(added_auto['images_added_crop']).transpose([2,1,0]), 0)
        added_frames = added_auto['added_frames'][0]
        # added_weights = np.array(added_auto['added_weights'])
        mask_added_train = np.array(added_auto['masks_added_crop']).transpose([2,1,0])
        # mask_center_train = np.array(added_auto['masks_center_crop'])
        mask_no_neighbors_train = np.logical_not(np.array(added_auto['masks_neighbors_crop'])).transpose([2,1,0])
        if num_mask_channel == 1:
            mask_train = np.expand_dims(mask_added_train.astype(image_train.dtype), 0)
        elif num_mask_channel == 2:
            mask_train = np.stack((mask_added_train, mask_no_neighbors_train), axis=0).astype(image_train.dtype)

        if img_option == 'multi_frame':
            (_, Ly, Lx, n) = mask_train.shape
            x_train = np.zeros((num_frame, Ly, Lx, n), dtype = image_train.dtype)
            for i in range(n):
                temp_frames = added_frames[i].transpose([2,1,0]).transpose((2,0,1))
                num_useful = temp_frames.shape[0]
                if num_useful < num_frame:
                    x_train[:num_useful, :, :, i] = temp_frames
                else:
                    x_train[:, :, :, i] = temp_frames[:num_frame]
        elif img_option == 'avg':
            x_train = image_train
        else:
            raise(ValueError('"img_option" must be in {}'.format({'avg','multi_frame'})))

        if mask_option == 'nomask':
            binary_last_channel = 0
        elif mask_option == 'mask':
            x_train = np.concatenate([x_train, mask_train], axis=0)
            binary_last_channel = 0
        elif mask_option == 'bmask':
            x_train = np.concatenate([x_train, mask_train], axis=0)
            binary_last_channel = num_mask_channel
        elif mask_option == 'Xmask':
            x_train = np.concatenate([x_train, mask_train*image_train], axis=0)
            binary_last_channel = 0
        else:
            raise(ValueError('"mask_option" must be in {}'.format({'nomask','mask','Xmask','bmask'})))

        if x_train.ndim == 3:
            x_train = np.expand_dims(x_train, -1)
        x_train = x_train.transpose((3,2,1,0))
        list_x_train.append(x_train)
        (n, Lx, Ly, c) = x_train.shape
        y_train = y_train.squeeze()
        if y_train.ndim == 0:
            y_train = np.expand_dims(y_train, -1)
        list_y_train.append(y_train)

    # num_true = y_train.sum()
    # num_total = y_train.size
    # num_false = num_total-num_true
    # if num_false>num_true:
    #     labels_true = y_train.nonzero()[0]
    #     labels_false = np.logical_not(y_train).nonzero()[0]
    #     ratio = int(np.round(num_false/num_true))
    #     labels_false_select = labels_false[::ratio]
    #     labels_select = np.hstack([labels_true, labels_false_select])
    #     x_train = x_train[labels_select]
    #     y_train = y_train[labels_select]
        
    #%
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x_train, y_train, test_size=test_fraction)
    
    for cv in range(nvideo):
        x_train = np.concatenate([x for (i, x) in enumerate(list_x_train) if i!=cv], axis=0)
        y_train = np.concatenate([y for (i, y) in enumerate(list_y_train) if i!=cv])
        class_weight = cw.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
        
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], c, Lx, Ly)
            # x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (c, Lx, Ly)
        else:
            x_train = x_train.reshape(x_train.shape[0], Lx, Ly, c)
            # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (Lx, Ly, c)
        
        x_train = x_train.astype('float32')
        # x_test = x_test.astype('float32')

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        # print(x_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        y_train_c = to_categorical(y_train, num_classes)
        # y_test = to_categorical(y_test, num_classes)
        print('{:d}/{:d}={:.1f}% true samples'.format(y_train.sum(), y_train.size, y_train.mean()*100))

        # %%
        model = CNN_classifier(input_shape, num_classes)
        txt=open(os.path.join(save_dir,"model_summary.txt"),'w')
        model.summary(print_fn = lambda x: txt.write(x + '\n'))
        txt.close()    
        
        with tf.device('/gpu:0'):
            if augmentation:
                print('Using real-time data augmentation.')
                # This will do preprocessing and realtime data augmentation:
                datagen = ImageDataGenerator(
                    shear_range=0.3,
                    rotation_range=360,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=[0.8, 1.2],
                    horizontal_flip=True,
                    vertical_flip=True,
                    random_shuffle_channel=random_shuffle_channel,
                    binary_last_channel=binary_last_channel,
                    random_mult_range=[.5, 2]
                )
            
                # Compute quantities required for feature-wise normalization
                # (std, mean, and principal components if ZCA whitening is applied).
                datagen.fit(x_train) # , augment=True
            
                # Fit the model on the batches generated by datagen.flow().
                try:
                    model.fit_generator(datagen.flow(x_train, y_train_c, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, verbose=1, class_weight=class_weight)
                except:
                    class_weight = {i:w for i,w in enumerate(class_weight)}
                    model.fit_generator(datagen.flow(x_train, y_train_c, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, verbose=1, class_weight=class_weight)
            
            
            else:
                model.fit(x_train, y_train_c, batch_size=batch_size, epochs=epochs, verbose=1)
        
        model_history = model.history.history
        #% Optimize th_cnn
        y_test = model.predict(x_train, batch_size=batch_size, verbose=0)
        num_GT = sum(y_train)
        num_th = len(list_th_cnn)
        list_recall = np.zeros(num_th)
        list_precision = np.zeros(num_th)
        # list_f1 = np.zeros(num_th)
        for (ii,th_cnn) in enumerate(list_th_cnn):
            pred_valid = y_test[:,1] > th_cnn
            num_pred = sum(pred_valid)
            num_both = sum(np.logical_and(pred_valid,y_train))
            list_recall[ii] = num_both / num_GT
            list_precision[ii]  = num_both / num_pred
        list_f1 = 2/(1/list_recall + 1/list_precision)
        list_f1[np.isnan(list_f1)] = 0
        list_f1[np.isinf(list_f1)] = 0
        ii_best = np.argmax(list_f1)
        th_cnn_best = list_th_cnn[ii_best]
        print(list_th_cnn)
        print(list_f1)
        print('The best th_cnn is', th_cnn_best)
        
        #% Save model and weights
        model_name = 'cnn_model_cv' + str(cv) #str(datetime.datetime.now()).replace(' ', '-').replace(':', '-')
        model_json = model.to_json()
        json_path = os.path.join(save_dir, model_name + '.json')
        
        with open(json_path, "w") as json_file:
            json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
        
        print('Saved trained model at %s ' % json_path)
        
        
        model_path = os.path.join(save_dir, model_name + '.h5')
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)
        
        # save training and validation loss after each eopch
        if 'accuracy' in model_history.keys():
            accuracy = model_history['accuracy']
        elif 'acc' in model_history.keys():
            accuracy = model_history['acc']
        else:
            accuracy = model_history
        history = { "loss": model_history['loss'], "accuracy": accuracy,'th_cnn_best': th_cnn_best,                
                    "list_recall": list_recall, "list_precision": list_precision, "list_f1": list_f1}
        sio.savemat(os.path.join(save_dir,"training_output_cv{}.mat".format(cv)), history)

        # txt=open(os.path.join(save_dir,"optimal_th_cnn_{}.txt".format(name_video)),'w')
        # txt.write(str(th_cnn_best))
        # txt.close()    
