import skimage
import os
import numpy as np
import cv2
import utils
import random


class WadStackedDatasetVal(utils.Dataset):
	"""Generates the WAD dataset.
	"""

	def get_filelist(self,datadir,mode):
		np.random.seed(231)
		files = os.listdir(datadir)
		np.random.shuffle(files)
		filelist = []
		if mode == 'train':
			filelist = files[:-1000]
		elif mode == 'val':
			filelist = files[-1000:]
		return filelist



	def get_datadir(self):
		cwd = os.getcwd()
		idx = cwd.find('Mask_RCNN')
		root = cwd[:idx]
		ddir = os.path.join(root,'cvpr-2018-autonomous-driving')
		return ddir

	def __init__(self,class_map=None):
		self._image_ids = []
		self.image_info = []
		# Background is always the first class
		self.class_info = [{"source": "wad", "id": 0, "name": "background"}]
		self.source_class_ids = {}
		self.datadir = self.get_datadir()
		self.object_map = {'car':1,'motorcycle':2,'bicycle':3,'person':4,
							'truck':5,'bus':6,'tricycle':7,}
		x_train_dir = os.path.join(self.datadir, 'train_stacked')
		self.filelist = self.get_filelist(x_train_dir,'val')

	def get_masks(self,label_im):
		"""Return lists of binary masks and ids corresponding to instances.
	Params:
	    label_im - label mask image from WAD as 2D numpy array
	Returns:
	    ids - list of ids corresponding to instance masks
	    masks - list of binary masks, each corresponding to one instance
	"""
		instances = np.unique(label_im)
		ids = []
		masks = []

		wad_to_ours = {
			33: 1,
			34: 2,
			35: 3,
			36: 4,
			38: 5,
			39: 6,
			40: 7
		}

		for instance in instances:
			if instance != 255 and instance != 65535:
				wad_id = int(instance / 1000)
				if wad_id not in wad_to_ours:
					continue
				our_id = wad_to_ours[wad_id]
				mask = (label_im == instance)
				# mask = mask.astype(np.int8)
				ids.append(our_id)
				masks.append(mask)
		if len(ids) == 0:
			return [0], np.zeros((label_im.shape[0], label_im.shape[1], 1)).astype(bool)
		# print("NUM INSTANCES: ", len(ids), len(masks), 'MASK SIZE: ', masks[0].shape)
		return ids, np.stack(masks,axis=2)

	def load_wad(self):
		"""Generate the requested number of synthetic images.
		count: number of images to generate.
		height, width: the size of the generated images.
		"""

		x_train_dir = os.path.join(self.datadir, 'train_stacked')
		y_train_dir = os.path.join(self.datadir, 'train_label')

		# Add classes
		for objName,class_num in self.object_map.items():
			 self.add_class("wad", class_num, objName)

		# Add images
		image_id = 0
		for filename in self.filelist:
			#print('FILE train', filename)
			if filename.endswith(".jpg"): 
				rootname = filename[7:-4]
				#print(filename)
				xfilepath = os.path.join(x_train_dir,filename)
				yfilepath = os.path.join(y_train_dir,rootname+'_instanceIds.png')
				self.add_image('wad',image_id = image_id,path = xfilepath, xfilepath = xfilepath, rootname = rootname, yfilepath = yfilepath)
				image_id = image_id+1

	def load_image(self, image_id):
		"""Generate an image from the specs of the given image ID.
		Typically this function loads the image from a file, but
		in this case it generates the image on the fly from the
		specs in image_info.
		"""
		info = self.image_info[image_id]
		img_path = info['path']
		#print(img_path)
		image = skimage.io.imread(img_path)

		return image

	def image_reference(self, image_id):
		"""Return the wad data of the image."""
		info = self.image_info[image_id]
		if info["source"] == "wad":
			return info
		else:
			super(self.__class__).image_reference(self, image_id)

	def load_mask(self, image_id):
		"""Generate instance masks for shapes of the given image ID.
		"""
		info = self.image_info[image_id]
		img_path = info['xfilepath']
		img_rootname = info['rootname']
		segm_img_path = info['yfilepath']
		segm_img = skimage.io.imread(segm_img_path)

		class_ids,segm_masks = self.get_masks(segm_img)
		# print("SEGM_MASKS SHAPE", segm_masks.shape)
		# Map class names to class IDs.
		return segm_masks, np.array(class_ids).astype(np.int32)

#dataset_train = WadDataset()
#dataset_train.load_wad()
#dataset_train.prepare()
