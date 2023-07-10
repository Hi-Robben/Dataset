# -*- coding: utf-8 -*-

import os
import zipfile

import keras.utils

from distinguish_attack.genVal import DataGenerator1
from distinguish_attack.generate import DataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras import optimizers

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
#Loading a dataset for train
class DA_Dataset:
    def __init__(self, imple):
        # imple_path = './dataset/'+imple
        # if not os.path.exists(imple_path):
        #     with zipfile.ZipFile(imple_path+'.zip') as z:
        #         z.extractall(imple_path)

        # tmp0 = np.load(imple_path +'/train/fixed.npy',  mmap_mode='r')
        # tmp1 = np.load(imple_path +'/train/random.npy', mmap_mode='r')

        ## 首次四分类 5000条  结果正常
        # tmp0 = np.load('./data/four/data0.npy', mmap_mode='r')
        # print(tmp0.shape)
        # tmp1 = np.load('./data/four/data1.npy', mmap_mode='r')
        # print(tmp1.shape)
        # tmp2 = np.load('./data/four/data2.npy', mmap_mode='r')
        # print(tmp2.shape)
        # tmp3 = np.load('./data/four/data3.npy', mmap_mode='r')


        tmp0 = np.load('./data/1k_64class/data0.npy', mmap_mode='r')
        tmp1 = np.load('./data/1k_64class/data1.npy', mmap_mode='r')
        x_tmp = np.append(tmp0, tmp1, axis=0)
        for i in range(2, 32):
            tmp2 = np.load('./data/1k_64class/data{}.npy'.format(i), mmap_mode='r')
            x_tmp = np.append(x_tmp, tmp2, axis=0)

        # tmp0 = np.load('./data/16class/data0.npy', mmap_mode='r')
        # tmp1 = np.load('./data/16class/data1.npy', mmap_mode='r')
        # tmp2 = np.load('./data/16class/data2.npy', mmap_mode='r')
        # tmp3 = np.load('./data/16class/data3.npy', mmap_mode='r')
        # tmp4 = np.load('./data/16class/data4.npy', mmap_mode='r')
        # tmp5 = np.load('./data/16class/data5.npy', mmap_mode='r')
        # tmp6 = np.load('./data/16class/data6.npy', mmap_mode='r')
        # tmp7 = np.load('./data/16class/data7.npy', mmap_mode='r')
        # tmp8 = np.load('./data/16class/data8.npy', mmap_mode='r')
        # tmp9 = np.load('./data/16class/data9.npy', mmap_mode='r')
        # tmp10 = np.load('./data/16class/data10.npy', mmap_mode='r')
        # tmp11 = np.load('./data/16class/data11.npy', mmap_mode='r')
        # tmp12 = np.load('./data/16class/data12.npy', mmap_mode='r')
        # tmp13 = np.load('./data/16class/data13.npy', mmap_mode='r')
        # tmp14 = np.load('./data/16class/data14.npy', mmap_mode='r')
        # tmp15 = np.load('./data/16class/data15.npy', mmap_mode='r')

        # tmp0 = np.load('./data/eight/data20.npy', mmap_mode='r')
        # tmp1 = np.load('./data/eight/data21.npy', mmap_mode='r')
        # tmp2 = np.load('./data/eight/data22.npy', mmap_mode='r')
        # tmp3 = np.load('./data/eight/data23.npy', mmap_mode='r')
        # tmp0 = np.load('./sum1/sum1.npy', mmap_mode='r')

        wave_num = tmp0.shape[0]
        self.in_size = tmp0.shape[1]

        train_size = 20000 # wave_num * 2 - 10000
        test_size = 12000

        # minmax_scaler = preprocessing.MinMaxScaler()
        # x_tmp = tmp0/256.
        # # x_tmp = minmax_scaler.fit_transform(x_tmp)
        # minmax_scaler = preprocessing.MinMaxScaler()

        # x_tmp=np.append(tmp0, tmp1, axis=0)
        # print(x_tmp.shape)
        # x_tmp=np.append(x_tmp, tmp2, axis=0)
        # print(x_tmp.shape)
        # x_tmp=np.append(x_tmp, tmp3, axis=0)
        # print(x_tmp.shape)
        # x_tmp = np.append(x_tmp, tmp4, axis=0)
        # x_tmp = np.append(x_tmp, tmp5, axis=0)
        # x_tmp = np.append(x_tmp, tmp6, axis=0)
        # x_tmp = np.append(x_tmp, tmp7, axis=0)
        # x_tmp = np.append(x_tmp, tmp8, axis=0)
        # x_tmp = np.append(x_tmp, tmp9, axis=0)
        # x_tmp = np.append(x_tmp, tmp10, axis=0)
        # x_tmp = np.append(x_tmp, tmp11, axis=0)
        # x_tmp = np.append(x_tmp, tmp12, axis=0)
        # x_tmp = np.append(x_tmp, tmp13, axis=0)
        # x_tmp = np.append(x_tmp, tmp14, axis=0)
        # x_tmp = np.append(x_tmp, tmp15, axis=0) / 256.
        x_tmp = x_tmp/256.
        print(np.array(x_tmp).shape)
        # x_tmp = minmax_scaler.fit_transform(x_tmp)

        # train_datagen = ImageDataGenerator(rescale=1. / 255)
        # test_datagen = ImageDataGenerator(rescale=1. / 255)
        # train_generator = train_datagen.flow_from_directory(
        #     train_dir,
        #     target_size=(150, 150),
        #     batch_size=20,
        #     class_mode='binary')
        # validation_generator = test_datagen.flow_from_directory(
        #     validation_dir,
        #     target_size=(150, 150),
        #     batch_size=20,
        #     class_mode='binary')

        # y_tmp=np.array([0]*wave_num + [1]*wave_num + [2]*wave_num + [3]*wave_num)

        # y_tmp = np.array([0] * wave_num + [1] * wave_num + [2] * wave_num + [3] * wave_num + [4] * wave_num + [5] * wave_num + [6] * wave_num + [7] * wave_num)
        y_tmp = [0] * wave_num
        for i in range(1, 32):
            y_tmp = y_tmp + [i] * wave_num
        # y_tmp = np.array([0]*wave_num + [1]*wave_num + [2]*wave_num + [3]*wave_num + [4]*wave_num + [5]*wave_num + [6]*wave_num + [7]*wave_num + [8]*wave_num + [9]*wave_num + [10]*wave_num + [11]*wave_num + [12]*wave_num + [13]*wave_num + [14]*wave_num + [15]*wave_num)
        y_tmp = np.array(y_tmp)
        print(y_tmp.shape)
        # y_tmp = keras.utils.to_categorical(y_tmp)
        # y_tmp = np.array([0] * wave_num + [1] * wave_num)

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_tmp, y_tmp, train_size = train_size, test_size = test_size, stratify=y_tmp, random_state=0)


# Neural Network Configuration
class DA_Net:
    def __init__(self, in_size, type):
        input_shape = (in_size,1)
        input_w = tf.keras.layers.Input(shape=input_shape)

        if type == "vgg":
            print("vgg start...")
            w = tf.keras.layers.Conv1D(4, 3, activation='relu', padding='same')(input_w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(4, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)

            w = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)

            w = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)

            w = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)

            w = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)

        # Configuration of a convolutional neural network used to destinguish fixed/random waves of each implementation without masked AES software
        if type == 'cnn':
            w = tf.keras.layers.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(input_w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)
            w = tf.keras.layers.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)
            w = tf.keras.layers.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)
            w = tf.keras.layers.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.AveragePooling1D(2, strides=2)(w)

        # Configuration of an all-connected layered neural network used to destinguish fixed/random waves of masked AES software
        elif type == 'fc':
            w = tf.keras.layers.Flatten()(input_w)
            w = tf.keras.layers.Dense(32, kernel_initializer='he_uniform')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)
            w = tf.keras.layers.Dense(32, kernel_initializer='he_uniform')(w)
            w = tf.keras.layers.BatchNormalization()(w)
            w = tf.keras.layers.Activation('selu')(w)

        # Same settings for all implementations
        w = tf.keras.layers.Flatten()(w)
        w = tf.keras.layers.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
        w = tf.keras.layers.BatchNormalization()(w)
        w = tf.keras.layers.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
        # w = tf.keras.layers.Dropout(0.5)(w)
        
        output = tf.keras.layers.Dense(32, activation='softmax')(w)
        self.model = tf.keras.models.Model(input_w,output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# To train a network 
def train(imple, type, epoch):
    tf.random.set_seed(0)

    train_dir = "./data/split_64class"
    # Loading dataset
    data = DA_Dataset(imple)
    # train_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # train_generator = train_datagen.flow_from_directory(
    #     train_dir,
    #     save_format="npy",
    #     target_size=(20, 100000),
    #     batch_size=20,
    #     class_mode='binary')
    # validation_generator = test_datagen.flow_from_directory(
    #     train_dir,
    #     save_format="npy",
    #     target_size=(20, 100000),
    #     batch_size=20,
    #     class_mode='binary')

    # 读取样本名称，然后根据样本名称去读取数据
    # class_num = 0
    # train_datas = []
    # for file in os.listdir("./data/split_64class/train"):
    #     file_path = os.path.join("./data/split_64class/train", file)
    #     if os.path.isdir(file_path):
    #         class_num = class_num + 1
    #         for sub_file in os.listdir(file_path):
    #             train_datas.append(os.path.join(file_path, sub_file))
    #
    # training_generator = DataGenerator(train_datas)
    #
    # test_datas = []
    # for file in os.listdir("./data/split_64class/val"):
    #     file_path = os.path.join("./data/split_64class/val", file)
    #     if os.path.isdir(file_path):
    #         class_num = class_num + 1
    #         for sub_file in os.listdir(file_path):
    #             test_datas.append(os.path.join(file_path, sub_file))
    #
    # test_generator = DataGenerator1(test_datas)

    # print(train_generator)
    #Configuration network
    net = DA_Net(data.in_size, type)
    # net = VGG19(weights=None, classes=64)
    # net = DA_Net(1, type)

    # Configuration for saving the model.
    metric = 'val_accuracy'
    modelCheckpoint = ModelCheckpoint(filepath = './model/'+imple+'.h5',
                                    monitor=metric,
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='max')

    # Running learning
    history = net.model.fit(data.x_train, data.y_train, \
        validation_data=(data.x_val, data.y_val), epochs=epoch, \
        batch_size=4, callbacks=[modelCheckpoint],verbose=1)
    # history = net.model.fit_generator(
    #     training_generator,
    #     steps_per_epoch=100,
    #     epochs=epoch,
    #     # validation_data=validation_generator,
    #     validation_steps=50,
    #     callbacks=[modelCheckpoint],
    #     verbose=1
    # )
    #
    # history = net.model.fit_generator(training_generator, epochs=100, validation_data=test_generator,
    #                                   validation_steps=50, max_queue_size=10, callbacks=[modelCheckpoint], workers=1)

    return max(history.history['val_accuracy'])