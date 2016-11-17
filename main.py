from voc_utils import list_image_sets,imgs_from_category_as_list,cat_name_to_cat_id,load_data_multilabel
from ImageObject import Image
from cnn import Architectures
import random
import numpy as np
import sys

img_categories = list_image_sets()

imgsize = (227,227)

train_images = []
val_images = []

df = load_data_multilabel('train')
data = df.as_matrix()
print(data[0,1:].tolist())

for row in data:
	imgobj = Image(row[0],imgsize[0],imgsize[1],row[1:].tolist())
	train_images.append(imgobj)

df = load_data_multilabel('val')
data = df.as_matrix()
print(data[0,1:].tolist())

for row in data:
	imgobj = Image(row[0],imgsize[0],imgsize[1],row[1:].tolist())
	val_images.append(imgobj)

del df
del data
'''
totalimage = np.zeros((3,227,227))
index=0
for img in train_images:
	index+=1
	x,f = img.readData()
	totalimage+=x
		
	if index%100==0:
		print(index)

totalimage/=float(len(train_images))
print(np.mean(totalimage[0,:,:]))
print(np.mean(totalimage[1,:,:]))
print(np.mean(totalimage[2,:,:]))

sys.exit(0)
'''

def train_image_generator(batch_size=5):
	while True:
		inputcount = 0
		features = []
		target = []
		random.shuffle(train_images)
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

def val_image_generator():
	while True:
		features = []
		target = []
		for img in val_images:
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

arch = nnobj.alexnet()
#arch = nnobj.vgg16()
#arch = nnobj.vgg19()

traingen = train_image_generator(batch_size=20)
valgen = val_image_generator()

arch.fit_generator(traingen,samples_per_epoch=len(train_images)-17,validation_data=valgen ,nb_val_samples=len(val_images),nb_epoch=20)
