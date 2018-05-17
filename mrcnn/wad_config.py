from config import Config

class WadConfig(Config):
    """Configuration for training on the Amazon Shelf Dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "wad"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 8  # background + 7 objects

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 704

    DETECTION_MIN_CONFIDENCE = 0.7

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 20
    DETECTION_MAX_INSTANCES = 20

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 80

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20

    USE_MINI_MASK = False
    
#config = ShelfConfig()
#config.display()