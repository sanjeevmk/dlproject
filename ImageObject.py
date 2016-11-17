import cv2
import numpy as np
from voc_utils import load_img
from skimage.transform import resize
import skimage

class Image:
	def __init__(self,filename,img_rows,img_cols,targetvector=None):
		self.imgname = filename
		self.rows = img_rows
		self.cols = img_cols

		#will be NONE for test data
		self.targetvector = targetvector

	def readData(self):
		img = load_img(self.imgname)

		# do image processing below
		#img = skimage.img_as_ubyte(img,force_copy=True)
		img = cv2.resize(img,(self.rows,self.cols))
		img = img.transpose((2,0,1))

		img[0,:,:] -= 103.574248976
		img[1,:,:] -= 111.753445311
		img[2,:,:] -= 116.547333502

		return img,self.targetvector
