"""
One-Shot Texture Segmentation
Implementd by Keras

"""

import numpy as np
import cv2
import sys
import imgaug

import os

from config import Config
from DTD_Dataset import DTD_Dataset
import texture_segmentation_network as texture_model

import keras.backend as K
import tensorflow as tf
from tensorflow.python import debug as tf_debug

ROOT_DIR = os.getcwd()
DATASET_PATH = os.path.expanduser('~') + "/Dataset/dtd/"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class TextureConfig(Config):
    NAME = "Texture"
    TRAIN_BN = True

if __name__ == '__main__':

    tf_config = tf.ConfigProto()
    tf_sess = tf.Session(config=tf_config)
    K.set_session(tf_sess)

    config = TextureConfig()

    model = texture_model.TextureNet(mode="inference", config=config,
                                     model_dir=DEFAULT_LOGS_DIR)

    exclude_layers = []
    # Find last trained weights
    # model_path = model.find_last()[1]
    # print ('model path: ', model_path)
    model_path = './FCN_DenseNet_texture_1600.h5'
    model.load_weights(model_path, by_name=False, exclude=exclude_layers)
    print ('successed load model')


    images = cv2.imread('./results/image_1530.jpg')
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    tile_images = cv2.imread('./results/texture_1530.jpg')
    tile_images = cv2.cvtColor(tile_images, cv2.COLOR_BGR2RGB)

    images = cv2.resize(images, (256,256))
    tile_images = cv2.resize(images,(64,64))

    images = [images]
    tile_images = [tile_images]
    results = model.segmentation(images, tile_images)

    cv2.imwrite('tmp_results.png', results[0]['mask'])
    print (results[0]['mask'])
    print ('finish')
