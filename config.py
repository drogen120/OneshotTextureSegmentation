"""
Learning to simplify
Implementd by Keras

Licensed under the MIT License (see LICENSE for details)
"""

import math
import numpy as np


class Config(object):

    NAME = None #Override in sub-class
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 10
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    TEXTURE_SIZE = 64
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported
    MEAN_PIXEL = np.array([123.0, 123.0, 123.0])
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    TRAIN_BN = False
    # Network architecture
    NETWORK_PARM =[
        # kernel,stride,filter_num
        [5, 2, 48],
        [3, 1, 128],
        [3, 1, 128],
        [3, 2, 256],
        [3, 1, 256],
        [3, 1, 256],
        [3, 2, 256],
        [3, 1, 512],
        [3, 1, 1024],
        [3, 1, 1024],
        [3, 1, 1024],
        [3, 1, 1024],
        [3, 1, 512],
        [3, 1, 256],
        [4, 0.5, 256],
        [3, 1, 256],
        [3, 1, 128],
        [4, 0.5, 128],
        [3, 1, 128],
        [3, 1, 48],
        [4, 0.5, 48],
        [3, 1, 24],
        [3, 1, 1],
    ]

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        self.TEXTURE_SHAPE = np.array(
            [self.TEXTURE_SIZE, self.TEXTURE_SIZE, 3])

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
