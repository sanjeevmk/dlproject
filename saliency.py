from voc_utils import list_image_sets,imgs_from_category_as_list,cat_name_to_cat_id,load_data_multilabel
from MajorityImageObject import Image
from cnn import Architectures
import random
import numpy as np
import sys
import matplotlib.pylab as plt
import skimage
import pylab
from crf import CRF
import pickle

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

arch = nnobj.alexnet(train=False)

inputcount = 0
features = []
target = []
batch_size = 1

acc = np.zeros((20,4))
index= 0 

exact=0
for img in val_images:
	if index%100==0:
		print(index)
	index+=1
	feature,targetv = img.readData()
	features.append(feature)

	pred = arch.predict(np.array(features))	

	'''
	pred[pred>0.7] = 1
	pred[pred<0.7] = 0

	pred = pred.flatten()
	'''

	if np.argmax(pred)  == np.argmax(targetv):
		print("computing")
		grad = nnobj.grad_wrt_input(np.array(features))

		#indices = [i for i in range(len(targetv)) if targetv[i]==1]

		indices = [np.argmax(targetv)]
		for ind in indices:
			salmap = grad[ind,0,:,:,:]

			image,seg = img.readOriginalSeg()
			image = image.copy().astype(np.uint8)

			if (seg==None):
				continue

			with open('dumps/'+img.imgname+'_a.pkl','w') as f:
				pickle.dump(salmap,f)

			continue

			salmap = np.transpose(salmap,(1,2,0))


			salmap = np.abs(salmap)
			salmap = np.sum(salmap,axis=2)
			mn = np.min(salmap)
			mx = np.max(salmap)

			salmap = np.float32((salmap- mn)*1.0/(mx - mn))
			salmapwo =  np.array(salmap)

			salmap[salmap < np.mean(salmap)] = 0
			salmap[salmap >= np.mean(salmap)] = 1

		
			continue	
			binarysalmap = np.float32(np.zeros((salmapwo.shape[0],salmapwo.shape[1],2)))

			for i in range(salmap.shape[0]):
				for j in range(salmap.shape[1]):
					binarysalmap[i][j][0] = np.max(salmap)-salmap[i][j]
					binarysalmap[i][j][1] = salmap[i][j]

			c = CRF(imgsize[0],imgsize[1],2)
			salmapcrf = c.full(binarysalmap,image)

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
		
			resultname = "saliencymaps/"+img_categories[ind]+"_"+str(index)+".png"
			plt.savefig(resultname)
			#plt.show()
	features = []

print(exact)

'''
print(acc)
for categ in img_categories:
	print(len(imgs_from_category_as_list(categ,'val')))
'''
