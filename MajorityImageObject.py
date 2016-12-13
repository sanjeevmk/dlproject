import cv2
import numpy as np
from voc_utils import load_img,load_img_seg
from skimage.transform import resize
import skimage

from voc_utils import load_annotation,cat_name_to_cat_id

NUM_LABELS = 20

class Image:
	def __init__(self,filename,img_rows,img_cols,targetvector=None):
		self.imgname = filename
		self.rows = img_rows
		self.cols = img_cols

		anno = load_annotation(filename)
		objs = anno.findAll('object')

		largestarea = 0
		largestobect = ""
		for obj in objs:
			obj_names = obj.findChildren('name')
			for name_tag in obj_names:
				bbox = obj.findChildren('bndbox')[0]
				xmin = int(bbox.findChildren('xmin')[0].contents[0])
				ymin = int(bbox.findChildren('ymin')[0].contents[0])
				xmax = int(bbox.findChildren('xmax')[0].contents[0])
				ymax = int(bbox.findChildren('ymax')[0].contents[0])

				if (xmax-xmin)*(ymax-ymin) > largestarea:
					largestarea = (xmax-xmin)*(ymax-ymin)
					largestobject = name_tag.contents[0]

		catid = cat_name_to_cat_id(largestobject)
		
		#will be NONE for test data
		self.targetvector = [0]*(NUM_LABELS+1)

		self.targetvector[catid] = 0.5
		self.targetvector[-1] = 0.5

	def readData(self):
		img = load_img(self.imgname)
		img = img.copy().astype(np.uint8)

		# do image processing below
		#img = skimage.img_as_ubyte(img,force_copy=True)
		img = cv2.resize(img,(self.rows,self.cols))
		img = img.transpose((2,0,1))

		'''
		img[0,:,:] -= 104
		img[1,:,:] -= 112
		img[2,:,:] -= 117
		'''

		#self.targetvector = [float(1.0*l/sum(self.targetvector)) for l in self.targetvector]
		return img,self.targetvector

	def readOriginal(self):
		img = load_img(self.imgname)

		# do image processing below
		img = cv2.resize(img,(self.rows,self.cols))

		return img

	def readOriginalSeg(self):
		img = load_img(self.imgname)

		# do image processing below
		img = cv2.resize(img,(self.rows,self.cols))

		seg = load_img_seg(self.imgname)

		if (seg== None):
			return img, None

		return img,seg
