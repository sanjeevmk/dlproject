from voc_utils import list_image_sets,imgs_from_category_as_list,cat_name_to_cat_id,load_data_multilabel
from MajorityImageObject import Image
from cnn import Architectures
import random
random.seed(10)
import numpy as np
import sys

img_categories = list_image_sets()

imgsize = (227,227)

train_images = []
val_images = []

df = load_data_multilabel('train')
data = df.as_matrix()

for row in data:
	imgobj = Image(row[0],imgsize[0],imgsize[1],row[1:].tolist())
	train_images.append(imgobj)

df = load_data_multilabel('val')
data = df.as_matrix()

for row in data:
	imgobj = Image(row[0],imgsize[0],imgsize[1],row[1:].tolist())
	val_images.append(imgobj)
'''
all_images = train_images + val_images

trainl = int(0.9*len(all_images))
train_images = all_images[:trainl]
val_images = all_images[trainl:]
'''

del df
del data

def train_image_generator(batch_size=5):
	while True:
		random.shuffle(train_images)
		inputcount = 0
		features = []
		target = []
		for img in train_images:
			if inputcount < batch_size-1:
				feature,targetv = img.readData()
				features.append(feature)
				target.append(targetv)

				inputcount+=1
			else:
				inputcount=0
				feature,targetv = img.readData()
				features.append(feature)
				target.append(targetv)

				yield [np.array(features),np.array(target)]
				del features
				del target
				features = []
				target = []

def val_image_generator(batch_size=5):
	while True:
		inputcount = 0
		features = []
		target = []
		for img in val_images:
			if inputcount < batch_size-1:
				feature,targetv = img.readData()
				features.append(feature)
				target.append(targetv)

				inputcount+=1
			else:
				inputcount=0
				feature,targetv = img.readData()
				features.append(feature)
				target.append(targetv)

				yield [np.array(features),np.array(target)]
				del features
				del target
				features = []
				target = []

def test_image_generator(batch_size=1000):
	inputcount = 0
	features = []

	for img in test_images:
		if inputcount < batch_size-1:
			features.append(img.readData())
			inputcount+=1
		else:
			inputcount=0
			features.append(img.readData())
			yield np.array(features)
			del features
			del names
			features = []
			names = []
	if len(features)!=0:
		yield np.array(features)

nnobj = Architectures()

#arch = nnobj.alexnetcam()
arch = nnobj.alexnet_branches()
#arch = nnobj.vgg16()
#arch = nnobj.vgg19()

traingen = train_image_generator(batch_size=100)
valgen = val_image_generator(batch_size=50)

#arch.fit_generator(traingen,samples_per_epoch=len(train_images)-17,validation_data=valgen ,nb_val_samples=len(val_images)-17,nb_epoch=15)
arch.fit_generator(traingen,samples_per_epoch=len(train_images)-17,nb_epoch=10)
arch.save_weights('newweights/alexvoc_branchavg.h5',overwrite=True)
#arch.save_weights('newweights/alexvoc_cam.h5',overwrite=True)
