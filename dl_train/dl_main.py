# -*- coding: utf-8 -*-

import os
import time

import dl_train
import dl_test
import dl_nll_v1
import dl_nll_v2

import tensorflow as tf
#
# os.putenv("CUDA_VISIBLE_DEVICES","0")
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# import tensorflow as tf
# import keras
# # from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# # set_session(sess)
# keras.backend.clear_session() #清理session

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'  # 运行程序，都会占用gpu0全部资源
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" #设置当前使用的GPU设备为0,1号两个设备,名称依次为'/gpu:0'、'/gpu:1'。
#[0,1]和[1,0]排列的设备是不同的，排在前面的设备优先级高，运行程序的时候会优先使用。


# main function
def main():
    # List showing the prepared dataset and the configuration of the network to be used for training.
    imple_list = [("aes_nonprotect_hw",'vgg'),
                ("aes_nonprotect_sw",'cnn'),
                ("keccak_nonprotect_sw",'cnn'),
                ("aes_masked_hw",'cnn'),
                ("aes_masked_sw",'fc'),
                ("ntru_nonprotect_sw",'cnn')]

    # number of epoch
    epoch = 300

    for imple, type in imple_list:
        print('\nimplementation:',imple)
        print('Train & Validation')
        # va = dl_train.train(imple, type, epoch)

        print('\nTest')
        ta = dl_test.test(imple)
        print('val accuracy : ' + str(va)[:6] + ' | test accuracy : ' + str(ta)[:6])

        print('\nLikelihood comparison')
        
        # The code that was run to calculate the values published in the paper.
        #dl_nll_v1.nll(imple)  
        
        # The code with measures to prevent underflow.
        # dl_nll_v2.nll(imple)

        time.sleep(5)
        exit()

if __name__ == "__main__":
    main()
