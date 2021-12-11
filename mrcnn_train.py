import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
# import utils
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image
import keras

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.getcwd()

# ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_with_clip")
iter_num = 0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, os.path.join("logs_with_clip","polypmodels","modelname.h5"))

#COCO_MODEL_PATH = os.path.join(ROOT_DIR, os.path.join("pretrained","mask_rcnn_coco.h5"))

#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained","mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class PolypConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "polyp"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 320
    # IMAGE_MAX_DIM = 384
    IMAGE_MIN_DIM = 512#480
    IMAGE_MAX_DIM = 512#640

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
    #RPN_ANCHOR_SCALES = (8 * 3, 16 * 3, 32 * 3, 64 * 3, 128 * 3)

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = PolypConfig()
config.display()


class PolypDataset(utils.Dataset):

    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # Rewrite load_shapes to include your own categories
    # and added path, mask_path, yaml_path to self.image_info information
    def load_shapes(self, count, img_folder, mask_folder, imglist, dataset_root_path, yaml_folder):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("polyp", 1, "polyp")
        self.add_class("polyp", 2, "clip")
        #self.add_class("shapes", 3, "hole")

        for i in range(count):
            # get image height & width
            filestr = imglist[i].split(".")[0]
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_folder + "/" + imglist[i]
            yaml_path = yaml_folder + "/" + filestr + ".yaml"
            cv_img = cv2.imread(img_folder + "/" + imglist[i])
            self.add_image("polyp", image_id=i, path=img_folder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)


    # rewrite load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        #print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)

        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("polyp") != -1:
                labels_form.append("polyp")
            elif labels[i].find("clip") != -1:
                labels_form.append("clip")
            #elif labels[i].find("hole") != -1:
            #    labels_form.append("hole")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


dataset_root_path = "./"
img_folder = dataset_root_path + "../data/data_split/train"
mask_folder = dataset_root_path + "../data/data_split_mask/train"
yaml_folder = dataset_root_path + "../data/data_split_yaml/train"


imglist = os.listdir(img_folder)
count = len(imglist)

# preparation of train,val dataset 
dataset_train = PolypDataset()
dataset_train.load_shapes(count, img_folder, mask_folder, imglist, dataset_root_path,yaml_folder)
dataset_train.prepare()

# print("dataset_train-->",dataset_train._image_ids)

val_img_folder = dataset_root_path + "../data/data_split/validation"
val_mask_folder = dataset_root_path + "../data/data_split_mask/validation"
yaml_folder = dataset_root_path + "../data/data_split_yaml/validation"
val_imglist = os.listdir(val_img_folder)
val_count = len(val_imglist)

dataset_val = PolypDataset()
dataset_val.load_shapes(val_count, val_img_folder, val_mask_folder, val_imglist, dataset_root_path,yaml_folder)
dataset_val.prepare()

# print("dataset_val-->",dataset_val._image_ids)

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    # print(COCO_MODEL_PATH)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.


model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10000,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.


#model.train(dataset_train, dataset_val,
#            learning_rate=config.LEARNING_RATE / 10,
#            epochs=100,
#            layers="all")