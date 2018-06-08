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

from config import Config
import utils
import model as modellib
import visualize
from model import log
from wad_config import WadConfig
from wad_dataset import WadDataset
from wad_dataset_val import WadDatasetVal

# Root directory of the project

# TODO(stevenzc): how many epochs, multiply steps per epoch
num_epochs = 500
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#scenes = os.listdir(rootdir)[1:]


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("DOWNLOADING COCO MODEL")
    utils.download_trained_weights(COCO_MODEL_PATH)

print("Making config")
config = WadConfig()
config.display()


print("Making Training Wad Dataset")
dataset_train = WadDataset()
dataset_train.load_wad()
dataset_train.prepare()

print("Making Validation Wad Dataset")
dataset_val = WadDatasetVal()
dataset_val.load_wad()
dataset_val.prepare()

print("Creating Model")
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

init_with = "last"  # imagenet, coco, or last
# TODO: change this to last after some training (stevenzc)

print("Initializing with Coco")

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

print("RUN TRAINING ON MODEL")
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE, epochs=num_epochs,layers='heads')

# Recreate the model in inference mode
# inference_config=config
# print("Creating Inference Model")
# model = modellib.MaskRCNN(mode="inference",config=inference_config, model_dir=MODEL_DIR)

# # Get path to saved weights
# # Either set a specific path or find last trained weights
# # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()[1]
# print("Loading weights")
# # Load trained weights (fill in path to trained weights here)
# assert model_path != "", "Provide path to trained weights"
# print("Loading weights from ", model_path)
# model.load_weights(model_path, by_name=True)

# def visualization(model,dataset_val,inference_config):
#     print("Visualization (on random Test Image, Ground Truths)")
#     # Test on a random image
#     image_id = random.choice(dataset_val.image_ids)
#     original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
#     log("original_image", original_image)
#     log("image_meta", image_meta)
#     log("gt_class_id", gt_class_id)
#     log("gt_bbox", gt_bbox)
#     log("gt_mask", gt_mask)
#     visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,dataset_train.class_names,figsize=(8, 8))
#     print("Detecting for test image")
#     results = model.detect([original_image], verbose=1)
#     print("Visualization (on random Test Image, Predicted)")
#     r = results[0]
#     visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#     dataset_val.class_names, r['scores'])

# TODO(stevenzc): this won't work because we don' thave a display
# visualization(model,dataset_val,inference_config)

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
# print("Evaluating on validation set, just 10 images")
# image_ids = np.random.choice(dataset_val.image_ids, 10)
# APs = []
# for image_id in image_ids:
#     # Load image and ground truth data
#     image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#     modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
#     # Run object detection
#     results = model.detect([image], verbose=0)
#     r = results[0]
#     if(r['masks'].shape[0]==0):
#         print('NO MASKS PREDICTED')
#         APs.append(0)
#         continue
#     print(gt_mask.shape,r['masks'].shape,r["rois"].shape, r["class_ids"].shape, r["scores"].shape)
#     # Compute AP
#     AP, precisions, recalls, overlaps =\
#     utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#      r["rois"], r["class_ids"], r["scores"], r['masks'])
#     print(AP)
#     APs.append(AP)

# print("mAP: ", np.mean(APs))
#visualize(model,dataset_val,inference_config)



