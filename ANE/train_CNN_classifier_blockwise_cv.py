import sys
import numpy as np
import time
import os

import keras
from keras import backend as K
from sklearn.model_selection import train_test_split

import tensorflow as tf
from image_preprocessing_keras_3mask import ImageDataGenerator
# from CNN_classifier import CNN_classifier as CNN_classifier

import json as simplejson
# from keras.models import model_from_json
from sklearn.utils import class_weight as cw

import scipy.io as sio
import h5py

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config = config)

if __name__ == "__main__":
    # sys.argv = ['.py', 'classifier_res0', '2', 'mask', '0']
    list_img_option = {'avg','multi_frame'}
    list_mask_option = {'nomask','mask','Xmask','bmask'}
    classifier = sys.argv[1] # 'classifier_res1' #  
    num_frame = int(sys.argv[2]) # 1 # 
    mask_option = sys.argv[3] # 'mask_thb' # 
    random_shuffle_channel = bool(int(sys.argv[4])) # True
    num_mask_channel = int(sys.argv[5]) # 4 # 
    # start = time.time()
    if num_frame <= 1:
        img_option = 'avg'
        random_shuffle_channel = False
    else:
        img_option = 'multi_frame'
    exec('from CNN_classifier import CNN_' + classifier + ' as CNN_classifier')
    folder = 'classifiers_{}_{}\\{}_{}+{} frames'.format(
        img_option, mask_option, classifier, num_frame, num_mask_channel)
    if random_shuffle_channel:
        folder = folder + '_shuffle'
    #%
    batch_size = 16
    num_classes = 2
    epochs = 2000
    # test_fraction = 0.25
    augmentation = True
    # input image dimensions
    # Lx, Ly = 50, 50

    # %%the data, shuffled and split between train and test sets
    list_name_video = ['blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps']
    # list_radius = [8,10,8,6] # 
    list_rate_hz = [10,15,7.5,5] # 
    # list_decay_time = [0.4, 0.5, 0.4, 0.75]
    Dimens = [(120,120),(80,80), (88,88),(192,240)]
    list_nframes = [6000, 9000, 9000, 1500]
    ID_part = ['_part11', '_part12', '_part21', '_part22']

    ind_video = 0
    name_video = list_name_video[ind_video]
    dir_video = 'E:\\data_CNMFE\\{}_original_masks\\GT Masks\\add_new_blockwise'.format(name_video)
    list_Exp_ID = [name_video+x for x in ID_part]
    nvideo = len(list_Exp_ID) # number of videos used for cross validation

    list_th_cnn = np.arange(0.2, 0.81, 0.05)
    save_dir = os.path.join(dir_video,folder)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    list_x_train = []
    list_y_train = []
    for cv in range(nvideo):
        Exp_ID = list_Exp_ID[cv]
        added_auto = sio.loadmat(os.path.join(dir_video, Exp_ID+'_added_CNNtrain_blockwise.mat'))
        y_train = np.array(added_auto['list_valid'])
        image_train = np.expand_dims(np.array(added_auto['images_added_crop']), 0)
        added_frames = added_auto['added_frames'][0]
        # added_weights = np.array(added_auto['added_weights'])
        mask_added_train = np.array(added_auto['masks_added_crop'])
        # mask_center_train = np.array(added_auto['masks_center_crop'])
        mask_no_neighbors_train = np.logical_not(np.array(added_auto['masks_neighbors_crop']))
        if num_mask_channel == 1:
            mask_train = np.expand_dims(mask_added_train.astype(image_train.dtype), 0)
        elif num_mask_channel == 2:
            mask_train = np.stack((mask_added_train, mask_no_neighbors_train), axis=0).astype(image_train.dtype)

        if img_option == 'multi_frame':
            (_, Ly, Lx, n) = mask_train.shape
            x_train = np.zeros((num_frame, Ly, Lx, n), dtype = image_train.dtype)
            for i in range(n):
                temp_frames = added_frames[i].transpose((2,0,1))
                num_useful = temp_frames.shape[0]
                if num_useful < num_frame:
                    x_train[:num_useful, :, :, i] = temp_frames
                else:
                    x_train[:, :, :, i] = temp_frames[:num_frame]
        elif img_option == 'avg':
            x_train = image_train
        else:
            raise(ValueError('"img_option" must be in {}'.format(list_img_option)))

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
            raise(ValueError('"mask_option" must be in {}'.format(list_mask_option)))

        x_train = x_train.transpose((3,2,1,0))
        list_x_train.append(x_train)
        (n, Lx, Ly, c) = x_train.shape
        list_y_train.append(y_train.squeeze())

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
        class_weight = cw.compute_class_weight('balanced', np.unique(y_train), y_train)
        
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
        y_train_c = keras.utils.to_categorical(y_train, num_classes)
        # y_test = keras.utils.to_categorical(y_test, num_classes)

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
                model.fit_generator(datagen.flow(x_train, y_train_c,
                                                batch_size=batch_size),
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    class_weight=class_weight)
            
            
            else:
                model.fit(x_train, y_train_c,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
        
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
        ii_best = np.argmax(list_f1)
        th_cnn_best = list_th_cnn[ii_best]
        print(list_th_cnn)
        print(list_f1)
        print('The best th_cnn is', th_cnn_best)
        
        #% Save model and weights
        # save_dir = 'C:\\Users\\ss723\\caiman_data\\model\\'
        model_name = 'cnn_model_' + name_video + '_cv' + str(cv) #str(datetime.datetime.now()).replace(' ', '-').replace(':', '-')
        model_json = model.to_json()
        json_path = os.path.join(save_dir, model_name + '.json')
        
        with open(json_path, "w") as json_file:
            json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
        
        print('Saved trained model at %s ' % json_path)
        
        
        model_path = os.path.join(save_dir, model_name + '.h5')
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)
        
        # save training and validation loss after each eopch
        model_history = model.history.history
        if 'accuracy' in model_history.keys():
            accuracy = model_history['accuracy']
        elif 'acc' in model_history.keys():
            accuracy = model_history['acc']
        else:
            accuracy = model_history
        history = { "loss": model_history['loss'], "accuracy": accuracy,'th_cnn_best': th_cnn_best,                
                    "list_recall": list_recall, "list_precision": list_precision, "list_f1": list_f1}
        sio.savemat(os.path.join(save_dir,"training_output_{}_cv{}.mat".format(name_video, cv)), history)
        # f = h5py.File(os.path.join(model_path,"training_output_{}.mat".format(name_video)), "w")
        # f.create_dataset("loss", data=model.history['loss'])
        # f.create_dataset("accuracy", data=model.history['accuracy'])
        # # if use_validation:
        # #     f.create_dataset("val_loss", data=results.history['val_loss'])
        # #     f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
        # f.close()

        # txt=open(os.path.join(save_dir,"optimal_th_cnn_{}.txt".format(name_video)),'w')
        # txt.write(str(th_cnn_best))
        # txt.close()    
