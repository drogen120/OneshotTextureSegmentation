"""
One-Shot Texture Segmentation
Implementd by Keras

Licensed under the MIT License (see LICENSE for details)
Written by Wang Qianlong
"""

import numpy as np
import cv2
import glob
import scipy.io
import os
import datetime
import re
import multiprocessing
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from keras import metrics

import utils
import config

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)

class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)

def mold_image(images, config):
    return (images.astype(np.float32) - config.MEAN_PIXEL) / 127.0

def load_image_gt(dataset, config, augmentation=None):
    texture_num = np.random.randint(2,6)
    image_data, mask_data, class_data, texture = \
        dataset.generate_image_mask(config.IMAGE_MAX_DIM, texture_num,
                                    config.TEXTURE_SIZE)

    if augmentation:
        import imgaug
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

        # Store shapes before augmentation to compare
        image_data_shape = image_data.shape
        mask_shape = mask_data.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image_data = det.augment_image(image_data.astype(np.uint8))
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask_data = det.augment_image(mask_data.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image_data.shape == image_data_shape, "Augmentation shouldn't change image size"
        assert mask_data.shape == mask_shape, "Augmentation shouldn't change image size"

    # cv2.imwrite("./image.jpg", image_data)
    # cv2.imwrite("./mask.jpg", mask_data*255)
    return image_data, mask_data, class_data, texture

def data_generator(dataset, config, shuffle=True, augmentation=None, batch_size=1):
    if dataset is None:
        return None

    b = 0
    image_index = -1
    error_count = 0

    while True:
        try:
            image, mask, class_id, texture_img = load_image_gt(dataset, config,
                                                              augmentation=augmentation)
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32
                )
                batch_textures = np.zeros(
                    (batch_size,) + texture_img.shape, dtype=np.float32
                )
                batch_masks = np.zeros(
                    (batch_size,) + mask.shape, dtype=np.float32
                )

                batch_images[b] = mold_image(image.astype(np.float32), config)
                batch_textures[b] = mold_image(texture_img.astype(np.float32),
                                              config)
                batch_masks[b] = mask

                b += 1
                if b >= batch_size:
                    inputs = [batch_images, batch_textures]
                    outputs = batch_masks

                    yield inputs, outputs

                    # start a new batch
                    b = 0

        except(GeneratorExit, KeyboardInterrupt):
            raise
        except:
            error_count += 1
            print("error_count", error_count)
            if error_count > 5:
                raise


def VGG16_graph(input_tensor, use_bias=True, train_bn=True):
    scope_name = "VGG16"
    with tf.name_scope(scope_name) as sc:
        # Block 1
        conv1_1 = KL.Conv2D(64, (3, 3),name='conv1_1', activation='relu', padding='same')(input_tensor)
        conv1_2 = KL.Conv2D(64, (3, 3),name='conv1_2', activation='relu', padding='same')(conv1_1)
        bn1 = BatchNorm()(conv1_2, training=train_bn)
        pool1 = KL.MaxPooling2D(pool_size=(2, 2))(bn1)
        drop1 = KL.Dropout(0.5)(pool1)

        # Block 2
        conv2_1 = KL.Conv2D(128, (3, 3),name='conv2_1', activation='relu', padding='same')(drop1)
        conv2_2 = KL.Conv2D(128, (3, 3),name='conv2_2', activation='relu', padding='same')(conv2_1)
        bn2 = BatchNorm()(conv2_2, training=train_bn)
        pool2 = KL.MaxPooling2D(pool_size=(2, 2))(bn2)
        drop2 = KL.Dropout(0.5)(pool2)

        # Block 3
        conv3_1 = KL.Conv2D(256, (3, 3),name='conv3_1', activation='relu', padding='same')(drop2)
        conv3_2 = KL.Conv2D(256, (3, 3),name='conv3_2', activation='relu', padding='same')(conv3_1)
        conv3_3 = KL.Conv2D(256, (3, 3),name='conv3_3', activation='relu', padding='same')(conv3_2)
        conv3_4 = KL.Conv2D(256, (3, 3),name='conv3_4', activation='relu', padding='same')(conv3_3)
        bn3 = BatchNorm()(conv3_4, training=train_bn)
        pool3 = KL.MaxPooling2D(pool_size=(2, 2))(bn3)
        drop3 = KL.Dropout(0.5)(pool3)

        # Block 4
        conv4_1 = KL.Conv2D(512, (3, 3),name='conv4_1', activation='relu', padding='same')(drop3)
        conv4_2 = KL.Conv2D(512, (3, 3),name='conv4_2', activation='relu', padding='same')(conv4_1)
        conv4_3 = KL.Conv2D(512, (3, 3),name='conv4_3', activation='relu', padding='same')(conv4_2)
        conv4_4 = KL.Conv2D(512, (3, 3),name='conv4_4', activation='relu', padding='same')(conv4_3)
        bn4 = BatchNorm()(conv4_4, training=train_bn)
        pool4 = KL.MaxPooling2D(pool_size=(2, 2))(bn4)
        drop4 = KL.Dropout(0.5)(pool4)

        # Block 5
        conv5_1 = KL.Conv2D(512, (3, 3),name='conv5_1', activation='relu', padding='same')(drop4)
        conv5_2 = KL.Conv2D(512, (3, 3),name='conv5_2', activation='relu', padding='same')(conv5_1)
        conv5_3 = KL.Conv2D(512, (3, 3),name='conv5_3', activation='relu', padding='same')(conv5_2)
        conv5_4 = KL.Conv2D(512, (3, 3),name='conv5_4', activation='relu', padding='same')(conv5_3)
        bn5 = BatchNorm()(conv5_4, training=train_bn)
        pool5 = KL.MaxPooling2D(pool_size=(2, 2))(bn5)
        
        return conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

def res_block(input_tensor, kernel_size, filters, stage, 
                   use_bias=True, train_bn=True):
    """The res_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layres
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'

    with tf.name_scope(conv_name_base) as sc:
        x = KL.Conv2D(nb_filter1, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2c', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu', name='res' + str(stage) + '_out')(x)
        return x

def VGG_encode_graph(input_tensor,use_bias=True, train_bn=True):
    conv1_1 , conv2_1, conv3_1, conv4_1, conv5_1 = VGG16_graph(input_tensor, use_bias, train_bn)
    x = KL.Conv2D(512, (1,1), activation="relu", name="conv_size_16")(conv5_1)
    x = res_block(x, 3, [512,512,512], "size_16", use_bias, train_bn)
    x = KL.UpSampling2D(size=(2,2))(x)
    x = KL.Concatenate()([x, conv4_1])
    x = KL.Conv2D(512, (1,1), activation="relu", name="conv_size_32")(x)
    x = res_block(x, 3, [512,512,512], "size_32", use_bias, train_bn)
    x = KL.UpSampling2D(size=(2,2))(x)
    x = KL.Concatenate()([x, conv3_1])
    x = KL.Conv2D(256, (1,1), activation="relu", name="conv_size_64")(x)
    x = res_block(x, 3, [256,256,256], "size_64", use_bias, train_bn)
    x = KL.UpSampling2D(size=(2,2))(x)
    x = KL.Concatenate()([x, conv2_1])
    x = KL.Conv2D(128, (1,1), activation="relu", name="conv_size_128")(x)
    x = res_block(x, 3, [128,128,128], "size_128", use_bias, train_bn)
    x = KL.UpSampling2D(size=(2,2))(x)
    x = KL.Concatenate()([x, conv1_1])
    x = KL.Conv2D(128, (1,1), activation="relu", name="conv_size_256")(x)
    x = res_block(x, 3, [128,128,128], "size_256", use_bias, train_bn)
    encode_output = KL.Conv2D(64, (1, 1), padding='same',
                  name="encoding_output_conv", use_bias=use_bias)(x)
    return encode_output, conv1_1 , conv2_1, conv3_1, conv4_1, conv5_1

def decode_graph(input_tensor, skip_layers, use_bias=True, train_bn=True):

    conv_name_base = "decode_graph"
    with tf.name_scope(conv_name_base) as sc:
        downsample_16 = KL.AveragePooling2D(pool_size=(16,16))(input_tensor)
        downsample_32 = KL.AveragePooling2D(pool_size=(8,8))(input_tensor)
        downsample_64 = KL.AveragePooling2D(pool_size=(4,4))(input_tensor)
        downsample_128 = KL.AveragePooling2D(pool_size=(2,2))(input_tensor)
        x = KL.Conv2D(128, (1,1), activation="relu",
                      name="decode_conv_1")(skip_layers.pop())
        x = KL.Concatenate()([x, downsample_16])
        x = KL.Conv2D(64, (1,1), activation="relu",
                      name="decode_combine_conv_1")(x)
        x = res_block(x, 3, [64, 64, 64], "decode_res_1", use_bias, train_bn)
        for index, downsample_item in enumerate([downsample_32, downsample_64,
                                                 downsample_128, input_tensor]):
            x = KL.Concatenate()([KL.UpSampling2D(size=(2,2))(x),
                                  downsample_item])
            x = KL.Conv2D(129, (1,1), activation="relu",
                          name="decode_conv_"+str(index+2))(x)
            x = KL.Concatenate()([x, skip_layers.pop()])
            x = KL.Conv2D(64, (1,1), activation="relu",
                          name="decode_combine_conv_"+str(index+2))(x)
            x = res_block(x, 3, [64, 64, 64], "decode_res_" + str(index+2),
                          use_bias, train_bn)

        decode_output = KL.Conv2D(1, (1,1), name="decode_result")(x)
        output_mask = KL.Activation("sigmoid",
                             name="prob_output")(decode_output)
        return decode_output, output_mask

def build_encode_model(use_bias=True, train_bn=True):
    input_image = KL.Input(shape=[None, None, 3], name="input_encode_image")
    outputs = VGG_encode_graph(input_image, use_bias, train_bn)
    return KM.Model([input_image], outputs, name="encode_model")

class TextureNet():
    def __init__(self, mode, config, model_dir):
        assert mode in ['training', 'inference']

        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()

        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        assert mode in ['training', 'inference']

        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image"
        )

        input_texture = KL.Input(
            shape=config.TEXTURE_SHAPE.tolist(), name="input_texture"
        )

        if mode == "training":
            encode_model = build_encode_model(train_bn=config.TRAIN_BN)
            texture_features, _, _, _, _, _ = encode_model([input_texture])
            image_features, *conv_skip_layers = encode_model([input_image])
            texture_features = KL.Lambda(lambda t : tf.slice(t,
                                              [0,0,0,0],[config.BATCH_SIZE,4,4,64]))(texture_features)
            texture_filter = KL.Lambda(lambda t :
                                       tf.transpose(t,[1,2,3,0]))(texture_features)
            corelations = KL.Lambda(lambda t : tf.nn.conv2d(t[0], t[1], strides=[1,1,1,1], 
                                            padding='SAME'))([image_features, texture_filter])

            decoded_mask, output_mask = decode_graph(corelations, conv_skip_layers,
                                       train_bn=config.TRAIN_BN)

            inputs = [input_image, input_texture]
            outputs = output_mask

            model = KM.Model(inputs, outputs, name="TextureNet")
            model.summary()

        else:
            encode_model = build_encode_model(train_bn=config.TRAIN_BN)
            texture_features, _, _, _, _, _ = encode_model(input_texture)
            image_features, *conv_skip_layers = encode_model(input_image)
            texture_features = KL.Lambda(lambda t : tf.slice(t,
                                              [0,0,0,0],[config.BATCH_SIZE,4,4,64]))(texture_features)
            texture_filter = KL.Lambda(lambda t :
                                       tf.transpose(t,[1,2,3,0]))(texture_features)
            corelations = KL.Lambda(lambda t : tf.nn.conv2d(t[0], t[1], strides=[1,1,1,1], 
                                            padding='SAME'))([image_features, texture_filter])

            decoded_mask, output_mask = decode_graph(corelations, conv_skip_layers,
                                       train_bn=train_bn)

            inputs = [input_image, input_texture]
            outputs = output_mask

            model = KM.Model(inputs, outputs, name="TextureNet")

        return model

    def compile(self, learning_rate, momentum):
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        self.keras_model.compile(optimizer=optimizer,
                                 loss="mean_squared_error",
                                 metrics=[metrics.binary_accuracy])

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("FCN_DenseNet"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/FCN\_DenseNet\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "FCN_DenseNet_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None):

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]


        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       augmentation=augmentation,
                                       batch_size=self.config.BATCH_SIZE)
        vis_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       augmentation=augmentation,
                                       batch_size=self.config.BATCH_SIZE)

        evaluator = utils.Segmentation_Evaluator(vis_generator,
                                                 self.log_dir+"/evaluator/",
                                                 eval_num=100)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True,
                                        write_images=False),
            evaluator,
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            period=10,
                                            verbose=0, save_weights_only=True)
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=False,
            verbose=1,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images, tile_images):
        molded_images = []
        molded_tile_images = []
        windows = []
        for image in images:
            molded_image, window, scalse, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)

            molded_image = mold_image(molded_image, self.config)
            molded_images.append(molded_image)
            windows.append(window)

        for tile_image in tile_images:
            tile_image, _, _, _ = utils.resize_image(
                tile_image,
                min_dim=0,
                max_dim=128,
                padding=self.config.IMAGE_PADDING
            )
            molded_tile_image = mold_image(tile_image, self.config)
            molded_tile_images.append(molded_tile_image)

        molded_images = np.stack(molded_images)
        molded_tile_images = np.stack(molded_tile_images)
        windows = np.stack(windows)

        return molded_images, molded_tile_images, windows

    def segmentation(self, images, tile_images, verbose=0):
        assert self.mode == "inference", "Create model in inference mode!"
        assert len(images) == self.config.BATCH_SIZE, "len(image) must equal \
        to batch size"
        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        molded_images, molded_tile_images, windows = self.mold_inputs(images, tile_images)
        if verbose:
            log("molded_images", molded_images)
            log("molded_tile_images", molded_tile_images)

        result_mask_probs = \
                self.keras_model.predict([molded_images, molded_tile_images], verbose=0)

        results = []
        for i, result_mask in enumerate(result_mask_probs):
            results.append({"mask": result_mask})

        return results

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers
