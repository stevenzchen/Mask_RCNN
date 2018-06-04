import skimage.io
import numpy as np 
import os


cwd = os.getcwd()
idx = cwd.find('Mask_RCNN')
root = cwd[:idx]
ddir = os.path.join(root,'cvpr-2018-autonomous-driving')
train_color = os.path.join(ddir,'train_color')
for txt in reversed(os.listdir(ddir)):
	if txt[:4] == 'road':
		print(txt)
		txt_path = os.path.join(ddir,txt)
		with open(txt_path) as f:
			content = f.readlines()
			num_frames = len(content)
			for i,line in enumerate(content):

				fn = line[41:70]
				fn_path = os.path.join(train_color,fn)
				if not os.path.exists(fn_path):
					continue

				cur_img = skimage.io.imread(fn_path, as_gray=True)
				prev_img = None
				next_img = None

				if i == 0:
					prev_img = cur_img
				else:
					prev_img_path = os.path.join(train_color,content[i-1][41:70])
					prev_img = skimage.io.imread(prev_img_path, as_gray=True)
				if i == num_frames-1:
					next_img = cur_img
				else:
					next_img_path = os.path.join(train_color,content[i+1][41:70])
					next_img = skimage.io.imread(next_img_path, as_gray=True)

				stacked_frames = np.stack((prev_img,cur_img,next_img),axis=2)
				skimage.io.imsave(os.path.join(ddir,'train_stacked','stacked'+fn),stacked_frames)