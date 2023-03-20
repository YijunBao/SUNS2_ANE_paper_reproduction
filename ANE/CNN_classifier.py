import sys
import numpy as np
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Add, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.layers import Dropout, Activation, Flatten

import tensorflow as tf
from sklearn.utils import class_weight as cw

# %%
def CNN_classifier_res0(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
# %%
def CNN_classifier_res1(input_shape, num_classes, padding = 'same'):
    input = Input(input_shape)
    x = Conv2D(16, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(2):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(512)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
# %%
def CNN_classifier_res2(input_shape, num_classes, padding = 'valid'):
    input = Input(input_shape)
    x = Conv2D(16, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(2):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = 'same')(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(512)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
    
# %%
def CNN_classifier_res3(input_shape, num_classes, padding = 'same'):
    input = Input(input_shape)
    x = Conv2D(32, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(2):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(512)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
# %%
def CNN_classifier_res4(input_shape, num_classes, padding = 'same'):
    input = Input(input_shape)
    x = Conv2D(32, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(2):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(64)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
    
# %%
def CNN_classifier_res5(input_shape, num_classes, padding = 'same'):
    input = Input(input_shape)
    x = Conv2D(16, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(2):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(64)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
    
# %%
def CNN_classifier_res6(input_shape, num_classes, padding = 'same'):
    input = Input(input_shape)
    x = Conv2D(32, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(3):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(64)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
        
# %%
def CNN_classifier_res7(input_shape, num_classes, padding = 'same'):
    input = Input(input_shape)
    x = Conv2D(32, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(4):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(64)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
    
# %%
def CNN_classifier_res8(input_shape, num_classes, padding = 'same'):
    input = Input(input_shape)
    x = Conv2D(16, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(3):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(64)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
    
# %%
def CNN_classifier_res9(input_shape, num_classes, padding = 'same'):
    input = Input(input_shape)
    x = Conv2D(32, (3,3), kernel_initializer='he_normal', padding = padding)(input)
    x = Activation('relu')(x)

    for t in range(3):
        n = 32 * 2**t
        # shortcut = Conv2D(n, (1,1), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = Conv2D(n, (1,1), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        shortcut = MaxPooling2D((2, 2), padding = padding)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # Layer 1
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(2,2), kernel_initializer='he_normal', padding = padding)(x)
        # Layer 2
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(n, (3,3), strides=(1,1), kernel_initializer='he_normal', padding = padding)(x)
        # Add Residue
        x = Add()([x, shortcut])     
        x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(512)(x) 
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax')(x)
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    return model
    
    