import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from voc_utils import list_image_sets,imgs_from_category_as_list,cat_name_to_cat_id,load_data_multilabel
from MajorityImageObject import Image
import numpy as np
import sys
import skimage
import pylab
from crf import CRF
import pickle
import os
from scipy.ndimage.filters import gaussian_filter
EPS = 1e-8
def scores(pp, gg):
	gt = np.asarray(gg)
	pred = np.asarray(pp)
	gt = gt.flatten()
	pred = pred.flatten()
	fp = [(gt[i]==0 and pred[i]==1) for i in range(len(gt))]
	fp = np.sum(fp)

        fn = [(gt[i]==1 and pred[i]==0) for i in range(len(gt))]
        fn = np.sum(fn)

        tp = [(gt[i]==1 and pred[i]==1) for i in range(len(gt))]
        tp = np.sum(tp)

        tn = [(gt[i]==0 and pred[i]==0) for i in range(len(gt))]
        tn = np.sum(tn)

	precision = (1.0*tp)/(tp+fp+EPS)
	recall = (1.0*tp)/(tp+fn+EPS)
	fscore = 2* precision*recall/(precision+recall+EPS)
	return fscore, precision, recall

def crfsharpen(salmap, image, g_xy, g_comp, bil_xy, bil_rgb, bil_comp):


	binarysalmap = np.float32(np.zeros((salmap.shape[0],salmap.shape[1],2)))

	for i in range(salmap.shape[0]):
		for j in range(salmap.shape[1]):
			binarysalmap[i][j][0] = np.max(salmap)-salmap[i][j]
			binarysalmap[i][j][1] = salmap[i][j]

	c = CRF(imgsize[0],imgsize[1],2, g_xy, g_comp, bil_xy, bil_rgb, bil_comp)
	salmapcrf = c.full(binarysalmap,image)

	mn = np.min(salmapcrf)
	mx = np.max(salmapcrf)

	salmapcrf = np.uint8((salmapcrf - mn)*255/(mx - mn))
	salmapcrf[salmapcrf > np.mean(salmapcrf)] = 1
	salmapcrf[salmapcrf != 1] = 0
	return salmapcrf


outdir = 'test2lowerrgb' 

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


def fullscores(flag=0,g_xy=6, g_comp=3, bil_xy=50, bil_rgb=10, bil_comp=10, train_size=30):
	inputcount = 0
	features = []
	target = []
	batch_size = 1
	fscore_mean=0
	precision_mean = 0
	recall_mean = 0
	index2=0
	index= 0 

	for img in train_images:#[:train_size]:
		if index%1==0:
			print(index)
		index+=1

		my_file = 'dumps/'+img.imgname+'_a.pkl'

		if os.path.isfile(my_file):
				salmap = None
				with open('dumps/'+img.imgname+'_a.pkl','r') as f:
					salmap = pickle.load(f)

				salmap = np.reshape(salmap,(227,227,1))
				#salmap = np.transpose(salmap,(1,2,0))

				image, seg = img.readOriginal()
				image = image.copy().astype(np.uint8)
				print img.imgname
				if (seg==None):
					print "continued..."
					continue
				image = image.copy().astype(np.uint8)

				seg = seg.copy().astype(np.float32)
				seg = np.mean(seg, axis=2)
				seg[seg>0] = 1
				
				salmap = np.abs(salmap)
				salmap = np.sum(salmap,axis=2)
				mn = np.min(salmap)
				mx = np.max(salmap)

				salmap = np.float32((salmap- mn)*1.0/(mx - mn))
				salmapwo = np.array(np.float32(salmap))
				
				salmapblur = gaussian_filter(salmap, sigma=2)
				salmapblurwo = np.array(np.float32(salmapblur))

				salmap[salmap < np.mean(salmap)] = 0
				salmap[salmap > 0] = 1
				
				salmapblur[salmapblur < np.mean(salmapblur)] = 0
				salmapblur[salmapblur > 0] = 1
				salmapcrf, salmapblurcrf = 0,0

				if (flag==0):
					#salmapcrf = crfsharpen(salmap,image,g_xy, g_comp, bil_xy, bil_rgb, bil_comp)
					fscore, precision, recall = scores(salmap, seg)
					print fscore, precision, recall
					fscore_mean = (fscore_mean*index2 + fscore)/(index2+1.0)
					precision_mean = (precision_mean*index2 + precision)/(index2+1.0)
					recall_mean = (recall_mean*index2 + recall)/(index2+1.0)
					index2 = index2 +1
					print fscore_mean,precision_mean,recall_mean
				if (flag==1):
					#salmapblurcrf = crfsharpen(salmapblur,image,g_xy, g_comp, bil_xy, bil_rgb,bil_comp)
					fscore, precision, recall = scores(salmapblur, seg)
					print fscore, precision, recall
					fscore_mean = (fscore_mean*index2 + fscore)/(index2+1.0)
					precision_mean = (precision_mean*index2 + precision)/(index2+1.0)
					recall_mean = (recall_mean*index2 + recall)/(index2+1.0)
					index2 = index2 +1
					print fscore_mean,precision_mean,recall_mean
				"""				
				plt.figure(1)
				plt.clf()
				plt.axis('off')
				f, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(nrows=1, ncols=8)
				# f.clf()
				ax1.get_xaxis().set_ticks([])
				ax2.get_xaxis().set_ticks([])
				ax3.get_xaxis().set_ticks([])
				ax4.get_xaxis().set_ticks([])
				ax5.get_xaxis().set_ticks([])
				ax6.get_xaxis().set_ticks([])
				ax7.get_xaxis().set_ticks([])
				ax8.get_xaxis().set_ticks([])
				ax1.get_yaxis().set_ticks([])
				ax2.get_yaxis().set_ticks([])
				ax3.get_yaxis().set_ticks([])
				ax4.get_yaxis().set_ticks([])
				ax5.get_yaxis().set_ticks([])
				ax6.get_yaxis().set_ticks([])
				ax7.get_yaxis().set_ticks([])
				ax8.get_yaxis().set_ticks([])

				ax1.imshow(image)
				ax2.imshow(salmapwo,cmap="jet",alpha=1.0,interpolation='nearest')
				ax3.imshow(salmap,cmap="jet",alpha=1.0,interpolation='nearest')
				ax4.imshow(salmapcrf,cmap="jet",alpha=1.0,interpolation='nearest')
				ax5.imshow(salmapblurwo,cmap="jet",alpha=1.0,interpolation='nearest')
				ax6.imshow(salmapblur,cmap="jet",alpha=1.0,interpolation='nearest')
				ax7.imshow(salmapblurcrf,cmap="jet",alpha=1.0,interpolation='nearest')
				ax8.imshow(seg,cmap="jet",alpha=1.0,interpolation='nearest')
				#plt.imshow(image)
				#plt.imshow(salmap,cmap="cool",alpha=0.55,interpolation='nearest')
			
				resultname = outdir+"/"+img.imgname+"_" + ".png"
				plt.savefig(resultname)
				plt.close()
				#plt.show()
				"""
				
		features = []
	return fscore_mean,precision_mean,recall_mean

'''
num_iters = 20
with open("gridsearch.txt","w+") as f:
	for i in range (num_iters):
		print(i)
		g_xy =random.choice([1,3,6,20, 40, 60, 100, 150, 200])
		g_comp=random.choice([1,3,9,27, 70, 100, 150])
		bil_xy= random.choice([1,5,20,30, 50, 70, 100])
		bil_rgb=random.choice([1,5,10,20,50,70,100])
		bil_comp= random.choice([1,10,20,30,40,80,150])
		ans = fullscores(1,g_xy, g_comp, bil_xy,
		 bil_rgb, bil_comp, train_size=350)
		f.write("Blur: "+str(g_xy)+","+ str(g_comp)+","+str(bil_xy)+","+
			str(bil_rgb)+","+str(bil_comp)+","+str(ans))
'''

ans = fullscores(1,train_size=350)
print("Blur: "+ str(ans))
