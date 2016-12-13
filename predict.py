from voc_utils import list_image_sets,imgs_from_category_as_list,cat_name_to_cat_id,load_data_multilabel
from MajorityImageObject import Image
from cnn import Architectures
import random
import numpy as np
import sys

img_categories = list_image_sets()

imgsize = (224,224)

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

del df
del data

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

nnobj = Architectures()

#arch = nnobj.alexnet(train=False)
arch = nnobj.vgg16(train=False)

traingen = train_image_generator(batch_size=20)
valgen = val_image_generator()

inputcount = 0
features = []
target = []
batch_size = 1

acc = np.zeros((20,4))
index= 0 

exact = 0
for img in val_images:
	if index%100==0:
		print(index)
	index+=1
	feature,targetv = img.readData()
	features.append(feature)

	pred = arch.predict(np.array(features))	
	features = []

	'''
	pred[pred>0.5] = 1
	pred[pred<0.5] = 0

	pred = pred.flatten()
	'''
	'''
	print("Pred:",np.where(pred>0.8))
	print("Target:",np.where(targetv>0.8))
	'''
	if np.argmax(pred) == np.argmax(targetv):
		exact+=1
	'''
	for i in range(20):
		if pred[0][i]==0 and targetv[i]==0:	
			acc[i][0]+=1
		if pred[0][i]==1 and targetv[i]==0:	
			acc[i][1]+=1
		if pred[0][i]==0 and targetv[i]==1:	
			acc[i][2]+=1
		if pred[0][i]==1 and targetv[i]==1:	
			acc[i][3]+=1
	'''

print(exact)
print(float(1.0*exact/len(val_images)))
'''
print(acc)
for categ in img_categories:
	print(len(imgs_from_category_as_list(categ,'val')))
'''
