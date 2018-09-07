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

class DTD_Dataset(object):
    """
    Describable Textures Dataset (DTD)

    Generate ramdom textures and masks using DTD images.
    """

    def __init__(self, dataset_path, subset="train", shuffle=True):
        type_list = ["train", "validation", "test"]
        assert subset in type_list, "subset should in \
                chose from train, validation or test."
        self.dataset_path = dataset_path
        self.dataset_image_path = dataset_path + "/images/"
        self.subset = type_list.index(subset) + 1
        self.shuffle = shuffle
        self.data_info = {}
        self.num_data = 0
        self.cur_index = 0
        self.image_id_list = []

    def load_data(self):
        mat = scipy.io.loadmat(self.dataset_path + "/imdb/imdb.mat")
        # print(mat["images"][0,0])
        image_id, image_path, image_type, image_class = mat["images"][0,0]
        image_id = image_id[0]
        image_path = image_path[0]
        image_type = image_type[0]
        image_class = image_class[0]
        idx = (image_type == self.subset)
        self.data_info["image_ids"] = image_id[idx]
        self.data_info["image_paths"] = image_path[idx]
        self.data_info["image_classes"] = image_class[idx]
        self.num_data = self.data_info["image_ids"].shape[0]

    def get_data(self):
        if self.cur_index == 0:
            self.image_id_list = np.arange(self.num_data)
            if self.shuffle == True:
                np.random.shuffle(self.image_id_list)

        image_file = \
             self.dataset_image_path+self.data_info["image_paths"][self.image_id_list[self.cur_index]][0]
        class_id = self.data_info["image_classes"][self.image_id_list[self.cur_index]]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.cur_index = (self.cur_index + 1) % self.num_data
        return image, class_id

    def generate_random_masks(self, img_size=256, segmentation_regions=5, points=None):
        batch_size = 1
        xs, ys = np.meshgrid(np.arange(0, img_size), np.arange(0, img_size))

        if points is None:
            # n_points = np.random.randint(1, segmentation_regions + 1, size=batch_size)
            n_points = [segmentation_regions]
            points   = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]

        masks = []
        for b in range(batch_size):
            dists_b = [np.sqrt((xs - p[0])**2 + (ys - p[1])**2) for p in points[b]]
            voronoi = np.argmin(dists_b, axis=0)
            masks_b = np.zeros((img_size, img_size, segmentation_regions))
            for m in range(segmentation_regions):
                masks_b[:,:,m][voronoi == m] = 1
            masks.append(masks_b)
        return masks[0]

    def get_random_texture_by_class(self, class_id):
        idx = (self.data_info["image_classes"] == class_id)
        image_files = self.data_info["image_paths"][idx]
        image_path = self.dataset_image_path + np.random.choice(image_files,1)[0][0]
        texture = cv2.imread(image_path)
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        return texture
    
    def random_crop(self, image, crop_height, crop_width):
        if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
            x = np.random.randint(0, image.shape[1]-crop_width)
            y = np.random.randint(0, image.shape[0]-crop_height)
            return image[y:y+crop_height, x:x+crop_width, :]
        else:
            raise Exception('Crop shape exceeds image dimensions!')

    def generate_image_mask(self, img_size, class_nums, texture_size):
        used_classes = []
        generated_image = np.zeros([img_size, img_size, 3], dtype=np.uint8)
        generated_mask = self.generate_random_masks(img_size, class_nums)

        while len(used_classes) < class_nums:
            image, class_id = self.get_data()
            image = cv2.resize(image, (img_size, img_size))
            if class_id in used_classes:
                continue
            index = len(used_classes)
            mask = generated_mask[:,:,index]
            mask = np.expand_dims(mask, axis=-1)
            comb_img = image * mask
            comb_img = comb_img.astype(np.uint8)
            generated_image += comb_img
            used_classes.append(class_id)
            
        # index = np.random.randint(0,len(used_classes))
        # random_class_id = used_classes[index]
        # texture = self.get_random_texture_by_class(random_class_id)
        # texture = cv2.resize(texture, (img_size, img_size))
        texture = image.copy()
        texture = self.random_crop(texture, texture_size, texture_size)
        # cv2.imshow("mask", generated_image)
        # cv2.waitKey(0)
        return generated_image, \
               np.expand_dims(generated_mask[:,:,index], axis=-1), \
               used_classes[index], texture


if __name__ == '__main__':
    DATASET_PATH = os.path.expanduser('~') + "/Dataset/dtd/"
    dataset = DTD_Dataset(DATASET_PATH)
    dataset.load_data()
    image_data, mask_data, class_data, texture = \
        dataset.generate_image_mask(256, 5, 64)
    print(class_data)
    cv2.imshow("image", image_data)
    cv2.imshow("mask", mask_data)
    cv2.imshow("texture", texture)
    cv2.waitKey(0)



