"""
One-Shot Texture Segmentation
Implementd by Keras

Licensed under the MIT License (see LICENSE for details)
Written by Wang Qianlong
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
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 2

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train TextureNet')

    parser.add_argument('--continue_train', required=False,
                        type = bool, default = False,
                        metavar=None,
                        help="Continue Train")

    args = parser.parse_args()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf_sess = tf.Session(config=tf_config)
    K.set_session(tf_sess)

    config = TextureConfig()

    model = texture_model.TextureNet(mode="training", config=config,
                                     model_dir=DEFAULT_LOGS_DIR)

    exclude_layers = []
    if args.continue_train == True:
        # Find last trained weights
        model_path = model.find_last()[1]
        model.load_weights(model_path, by_name=True, exclude=exclude_layers)

    dataset_train = DTD_Dataset(DATASET_PATH)
    dataset_train.load_data()

    dataset_val = DTD_Dataset(DATASET_PATH)
    dataset_val.load_data()

    augmentation = imgaug.augmenters.Sometimes(0.5, [
    imgaug.augmenters.Fliplr(0.5),
    imgaug.augmenters.Flipud(0.5),
    ])

    model.train(dataset_train, dataset_val, learning_rate = 0.001,
                   epochs = 400, layers = 'all', augmentation=augmentation)
