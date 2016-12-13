from voc_utils import list_image_sets,imgs_from_category_as_list,cat_name_to_cat_id,load_data_multilabel
from MajorityImageObject import Image
from cnn import Architectures
import pickle
import random
random.seed(10)
import numpy as np
import sys
import matplotlib.pylab as plt
import skimage
import pylab
from crf import CRF
import theano.tensor.nnet.abstract_conv as absconv
import keras.backend as K
import cv2

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

def get_classmap(model, X, ind):
	global nametoid
	inc = model.layers[0].input
	name = 'convolution2d_'+str(ind+1)
	#namew = 'convolution2d_'+str(2*(ind+1))
	layerid = nametoid[name]
	#wlayerid = nametoid[namew]
	conv6 = model.layers[layerid].output
	#wei = model.layers[wlayerid].W
	#conv6 = conv6*wei
	#conv6_resized = K.resize_images(conv6,15,15,'th')
	conv6_resized = absconv.bilinear_upsampling(conv6, 15,batch_size=1,num_input_channels=1)
	conv6_resized = K.spatial_2d_padding(conv6_resized, padding=(1, 1), dim_ordering='th')
	get_cmap = K.function([inc], conv6_resized)
	return get_cmap([X])


nnobj = Architectures()

arch = nnobj.alexnet_branches(train=False)
#arch = nnobj.vgg16(train=False)

nametoid = {}
for i in range(len(arch.layers)):
	nametoid[arch.layers[i].name] = i


inputcount = 0
features = []
target = []
batch_size = 1

acc = np.zeros((20,4))
index= 0 

exact=0
for img in train_images:
	if index%100==0:
		print(index)
	index+=1
	feature,targetv = img.readData()
	features.append(feature)

	pred = arch.predict(np.array(features)).tolist()
	pred = pred[0][:-1]
	targetv = targetv[:-1]

	'''
	pred[pred>0.7] = 1
	pred[pred<0.7] = 0

	pred = pred.flatten()
	'''

	if np.argmax(pred)  == np.argmax(targetv):
		print("computing")
		image,seg = img.readOriginalSeg()
		image = image.copy().astype(np.uint8)
		if (seg==None):
			continue

		#indices = [i for i in range(len(targetv)) if targetv[i]==1]

		indices = [np.argmax(targetv)]
		for ind in indices:
			salmap = get_classmap(arch,feature.reshape(1, 3, imgsize[0], imgsize[1]),ind)[0][0]

			with open('dumps/'+img.imgname+"_a.pkl",'w') as f:
				pickle.dump(salmap,f)

			continue
			print(salmap.shape)


			mn = np.min(salmap)
			mx = np.max(salmap)

			#salmap = np.float32((salmap- mn)*1.0/(mx - mn))
			salmapwo =  np.array(salmap)

			#salmap[salmap < np.max(salmap) - 0.1* np.max(salmap)] = 0
			#salmap[salmap >= np.max(salmap) - 0.1 * np.max(salmap)] = 1
			salmap[salmap < np.mean(salmap)] = 0
			salmap[salmap > 0] = 1

			binarysalmap = np.float32(np.zeros((salmap.shape[0],salmap.shape[1],2)))

			for i in range(salmap.shape[0]):
				for j in range(salmap.shape[1]):
					binarysalmap[i][j][0] = np.max(salmap)-salmap[i][j]
					binarysalmap[i][j][1] = salmap[i][j]

			image = cv2.resize(image,(227,227))
			c = CRF(227,227,2)
			salmapcrf = c.full(binarysalmap,image)

			mn = np.min(salmapcrf)
			mx = np.max(salmapcrf)

			salmapcrf = np.uint8((salmapcrf - mn)*255/(mx - mn))

			plt.figure(1)
			plt.clf()
			plt.axis('off')
			f, (ax1, ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4, num=1)
			# f.clf()
			ax1.get_xaxis().set_ticks([])
			ax2.get_xaxis().set_ticks([])
			ax3.get_xaxis().set_ticks([])
			ax4.get_xaxis().set_ticks([])
			ax1.get_yaxis().set_ticks([])
			ax2.get_yaxis().set_ticks([])
			ax3.get_yaxis().set_ticks([])
			ax4.get_yaxis().set_ticks([])

			ax1.imshow(image)
			ax2.imshow(salmapwo,cmap="jet",alpha=1.0,interpolation='nearest')
			ax3.imshow(salmap,cmap="jet",alpha=1.0,interpolation='nearest')
			ax4.imshow(salmapcrf,cmap="jet",alpha=1.0,interpolation='nearest')

			'''
			plt.imshow(image)
			plt.imshow(salmap,cmap="cool",alpha=0.55,interpolation='nearest')
			'''
		
			resultname = "branchavg/"+img_categories[ind]+"_"+str(index)+".png"
			plt.savefig(resultname)
			#plt.show()
	features = []

print(exact)

'''
print(acc)
for categ in img_categories:
	print(len(imgs_from_category_as_list(categ,'val')))
'''
