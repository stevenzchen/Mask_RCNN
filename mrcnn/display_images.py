import os
from model import log
import model as modellib
import visualize
from model import log
from wad_config import WadConfig
from wad_3frames_dataset import WadStackedDataset
from wad_3frames_dataset_val import WadStackedDatasetVal
import random
import numpy as np
import utils

ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
inference_config = WadConfig()
dataset_train = WadStackedDataset()
dataset_train.load_shelf()
dataset_train.prepare()
dataset_val = WadStackedDatasetVal()
dataset_val.load_shelf()
dataset_val.prepare()

model = modellib.MaskRCNN(mode="inference",config=inference_config, model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shelf_0200.h5")
#model_path = model.find_last()[1]
print("Loading weights from ", model_path)
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


def visualization(model,dataset_val,inference_config,img_id=0):
    print("Visualization (on random Test Image, Ground Truths)")
    # Test on a random image
    image_id = img_id
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,dataset_val.class_names,figsize=(8, 8))
    print("Detecting for test image")
    results = model.detect([original_image], verbose=1)
    print("Visualization (on random Test Image, Predicted)")
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
    dataset_val.class_names, r['scores'])

def vizz(image_id):
	visualization(model,dataset_val,inference_config,image_id)

visualization(model,dataset_val,inference_config)
