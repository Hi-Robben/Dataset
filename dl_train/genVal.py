# coding=utf-8
'''
Created on 2018-7-10
'''
# import keras
from tensorflow import keras
import math
import os
# import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class DataGenerator1(keras.utils.Sequence):

    def __init__(self, datas, batch_size=10, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
        self.x_val = self.data_generation(batch_datas=datas)
        self.y_val = self.data_generation(batch_datas=datas)

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)
        # self.x_val = X
        # self.y_val = y
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []

        # 生成数据
        for i, data in enumerate(batch_datas):
            # x_train数据
            # image = cv2.imread(data)
            image = np.load(data)
            # image = image.reshape(20, 5000)
            image = list(image)
            images.append(image)
            # y_train数据
            right = data.rfind("\\", 0)
            left = data.rfind("\\", 0, right) + 1
            class_name = data[left:right]
            # ret = np.zeros(16)
            # ret[int(class_name)] = 1
            # labels.append(ret)
            labels.append(int(class_name))
            # for i in range(20):
            #     labels.append(int(class_name))

            # if class_name == "dog":
            #     labels.append([0, 1])
            # else:
            #     labels.append([1, 0])
        # 如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        # print(images)

        # one_hot_train_labels = self.to_one_hot(labels)
        # # y = keras.utils.to_categorical(labels, num_classes=16)
        # x = np.array(images)
        # y = np.array(labels)
        # self.x_val = np.array(images)/256.
        # self.y_val = np.array(labels)
        return np.array(images)/256., np.array(labels)

    def to_one_hot(labels, dimension=16):

        results = np.zeros((len(labels), dimension))

        for i, label in enumerate(labels):
            results[i, label] = i

        return results


# # 读取样本名称，然后根据样本名称去读取数据
# class_num = 0
# train_datas = []
# for file in os.listdir("./data/split_64class"):
#     file_path = os.path.join("./data/split_64class", file)
#     if os.path.isdir(file_path):
#         class_num = class_num + 1
#         for sub_file in os.listdir(file_path):
#             train_datas.append(os.path.join(file_path, sub_file))

# # 数据生成器
# training_generator = DataGenerator(train_datas)
#
# # 构建网络
# model = Sequential()
# model.add(Dense(units=64, activation='relu', input_dim=100000))
# model.add(Dense(units=2, activation='softmax'))
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit_generator(training_generator, epochs=50, max_queue_size=10, workers=1)
