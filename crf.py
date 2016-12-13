import numpy as np
import pydensecrf.densecrf as dcrf


class CRF:
	def __init__(self,rows,cols,nlabels, g_xy=6, g_comp=3, bil_xy=50, bil_rgb=10, bil_comp=10):
		self.d = dcrf.DenseCRF2D(rows, cols, nlabels)  
		self.r = rows
		self.c = cols
		self.n = nlabels 
		self.g_xy= g_xy
		self.g_comp= g_comp
		self.bil_xy = bil_xy
		self.bil_rgb =bil_rgb
		self.bil_comp =bil_comp

	def unary(self,potentials):
		U = potentials     # Get the unary in some way.
		print(U.shape)        # -> (5, 640, 480)
		print(U.dtype)        # -> dtype('float32')
		U = U.reshape((self.n,-1)) # Needs to be flat.
		self.d.setUnaryEnergy(U)

	def commonpairwise(self,im):
		#self.d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

		# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
		# im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
		#self.d.addPairwiseBilateral(sxy=(5,5), srgb=(10,10,10), rgbim=im, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		self.d.addPairwiseGaussian(sxy=self.g_xy, compat=self.g_comp,kernel=dcrf.FULL_KERNEL, normalization=dcrf.NO_NORMALIZATION)
		self.d.addPairwiseBilateral(sxy=self.bil_xy, srgb=self.bil_rgb, rgbim=im, compat=self.bil_comp,kernel=dcrf.FULL_KERNEL,normalization=dcrf.NO_NORMALIZATION)
		#self.d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im, compat=10,kernel=dcrf.FULL_KERNEL,normalization=dcrf.NO_NORMALIZATION)

	def maxp(self):
		Q = self.d.inference(5)
		maxp = np.argmax(Q, axis=0).reshape((self.r,self.c))
		return maxp

	def full(self,potentials,im):
		self.unary(potentials)
		self.commonpairwise(im)
		return self.maxp()
