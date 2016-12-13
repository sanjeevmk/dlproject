from keras.models import Sequential,Model
from keras.optimizers import Adadelta,Adam,SGD
from keras.layers.core import Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D,MaxPooling2D,AveragePooling2D,UpSampling2D,ZeroPadding2D
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten,Dropout
from keras.layers.advanced_activations import LeakyReLU,PReLU,ELU,ParametricSoftplus,ThresholdedReLU,SReLU
from keras.callbacks import	EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge,Lambda
from keras.regularizers import ActivityRegularizer

from keras.layers.convolutional import Convolution1D,MaxPooling1D,AveragePooling1D,UpSampling1D
from customlayers import convolution2Dgroup, crosschannelnormalization, splittensor, Softmax4D
import theano
import numpy as np
import theano.tensor as T
from keras.optimizers import SGD,Adagrad
from keras.regularizers import l2,l1
import h5py

sgd = SGD(lr=1e-2,decay=0,nesterov=False,momentum=0.0)
adam = Adam(lr=1e-3)

def oquabloss(y_true,y_pred):
	return K.sum(K.log(1.0+K.exp(-1.0*y_true*y_pred)))

def maxnorm(vector):
	return 1.0*vector/K.max(vector)
	
def maxnormshape(shape):
	return shape

class Architectures:
	def vgg16(self,train=True):
		p = 0.5

		if not train:
			p = 0.0

		model = Sequential()
		model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
		model.add(Convolution2D(64, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(64, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(128, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(128, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))
		model.add(Flatten())
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(p))
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(p))
		model.add(Dense(1000, activation='softmax'))


		if train:
			model.load_weights('weights/vgg16_weights.h5')

		for layer in model.layers:
			layer.trainable = False

		for i in range(7):
			model.layers.pop()

		model.outputs = [model.layers[-1].output]
		model.layers[-1].outbound_nodes = []

		model.add(Convolution2D(1024,3,3,activation='relu',name='cam_conv1',border_mode='same'))
		model.add(AveragePooling2D((14,14)))
		model.add(Flatten())
		model.add(Dense(20,activation='sigmoid',name='dense_smk'))

		print(model.summary())
		if not train:
			model.load_weights('newweights/vggvoc_cam.h5')

	
		model.compile(loss='categorical_crossentropy',optimizer=sgd)

		return model
	
	def alexnet(self,train=True):
		p = 0.6
			
		if not train:
			p=0.0

		inputs = Input(shape=(3,227,227))
		self.inputs = inputs
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
													 name='conv_1')(inputs)

		conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
		conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
		conv_2 = ZeroPadding2D((2,2))(conv_2)
		conv_2 = merge([
				Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_2)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

		conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
		conv_3 = crosschannelnormalization()(conv_3)
		conv_3 = ZeroPadding2D((1,1))(conv_3)
		conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3'
)(conv_3)

		conv_4 = ZeroPadding2D((1,1))(conv_3)
		conv_4 = merge([
				Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_4)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

		conv_5 = ZeroPadding2D((1,1))(conv_4)
		conv_5 = merge([
				Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_5)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

		finalmax = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

		dense_1_1 = Flatten(name="flatten")(finalmax)
		dense_1 = Dense(4096, activation='relu')(dense_1_1)
		dense_2 = Dropout(p)(dense_1)
		dense_2 = Dense(4096, activation='relu')(dense_2)
		dense_3 = Dropout(p)(dense_2)  
		dense_4 = Dense(1000)(dense_3)
		prediction = Activation("softmax",name="softmax")(dense_4)

		model = Model(input=inputs, output=prediction)

		if train:
			model.load_weights('weights/alexnet_weights.h5')

		for i in range(2):
			model.layers.pop()
		'''
		for layer in model.layers:
			layer.trainable = False
		'''

		dense_3 = Dense(2048)(dense_3)
		dense_3 = Dropout(p)(dense_2)  
		dense_4 = Dense(20,init='glorot_uniform')(dense_3)

		prediction = Activation("softmax",name="softmax")(dense_4)

		model = Model(input=inputs, output=prediction)

		if not train:
			model.load_weights('newweights/alexvoc_drop40.h5')

		self.model = model

		print(model.summary())
		self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['mse'])
		return self.model

	def alexnetcam(self,train=True):
		p = 0.6
			
		if not train:
			p=0.0

		inputs = Input(shape=(3,227,227))
		self.inputs = inputs
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
													 name='conv_1')(inputs)

		conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
		conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
		conv_2 = ZeroPadding2D((2,2))(conv_2)
		conv_2 = merge([
				Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_2)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

		conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
		conv_3 = crosschannelnormalization()(conv_3)
		conv_3 = ZeroPadding2D((1,1))(conv_3)
		conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3'
)(conv_3)

		conv_4_0 = ZeroPadding2D((1,1))(conv_3)
		conv_4 = merge([
				Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_4_0)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

		conv_5_0 = ZeroPadding2D((1,1))(conv_4)
		conv_5 = merge([
				Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_5_0)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

		finalmax = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

		dense_1_1 = Flatten(name="flatten")(finalmax)
		dense_1 = Dense(4096, activation='relu')(dense_1_1)
		dense_2 = Dropout(p)(dense_1)
		dense_2 = Dense(4096, activation='relu')(dense_2)
		dense_3 = Dropout(p)(dense_2)  
		dense_4 = Dense(1000)(dense_3)
		prediction = Activation("softmax",name="softmax")(dense_4)

		model = Model(input=inputs, output=prediction)

		if train:
			model.load_weights('weights/alexnet_weights.h5')

		for i in range(9):
			model.layers.pop()

		for layer in model.layers:
			layer.trainable = False

		conv_5 = Convolution2D(256,3,3,activation="relu",border_mode='same',init='glorot_uniform')(conv_5_0)
		#conv_5 = Convolution2D(32,3,3,activation="relu",border_mode='same',init='glorot_uniform')(conv_5_0)
		finalmax = AveragePooling2D((15,15))(conv_5)
		dense_3 = Flatten()(finalmax)
		dense_4 = Dense(20,init='glorot_uniform')(dense_3)

		prediction = Activation("sigmoid",name="softmax")(dense_4)

		model = Model(input=inputs, output=prediction)

		if not train:
			model.load_weights('newweights/alexvoc_cam.h5')

		self.model = model

		print(model.summary())
		self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['mse'])
		return self.model

	def branchout(self,fmap,init):
		#bout = Convolution2D(32,3,3,activation="relu",border_mode='same',init=init)(fmap)
		bout = Convolution2D(1,4,4,activation="relu",border_mode='same',init=init)(fmap)
		bout = AveragePooling2D((15,15),dim_ordering='th')(bout)
		#bout = Convolution2D(1,15,15,border_mode='valid',init=init)(bout)
		#bout = MaxPooling2D((15,15),dim_ordering='th')(bout)
		#bout = Lambda(self.lse,output_shape=(1,1,1))(bout)
		bout = Flatten()(bout)
		return bout
		
	def lse(self,fmap):
		pool = (1.0/2.0)*K.log(K.mean(K.exp(2.0*fmap)))
		return K.reshape(pool,(1,1,1))

	def maxnorm(self,vector):
		return 1.0*vector/K.max(vector)
		
	def maxnormshape(self,shape):
		return shape
	
	def alexnet_branches(self,train=True):
		p = 0.6
			
		if not train:
			p=0.0

		inputs = Input(shape=(3,227,227))
		self.inputs = inputs
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
													 name='conv_1')(inputs)

		conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
		conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
		conv_2 = ZeroPadding2D((2,2))(conv_2)
		conv_2 = merge([
				Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_2)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

		conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
		conv_3 = crosschannelnormalization()(conv_3)
		conv_3 = ZeroPadding2D((1,1))(conv_3)
		conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3'
)(conv_3)

		conv_4_0 = ZeroPadding2D((1,1))(conv_3)
		conv_4 = merge([
				Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_4_0)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

		conv_5_0 = ZeroPadding2D((1,1))(conv_4)
		conv_5 = merge([
				Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1)
)(
						splittensor(ratio_split=2,id_split=i)(conv_5_0)
				) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

		finalmax = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

		dense_1_1 = Flatten(name="flatten")(finalmax)
		dense_1 = Dense(4096, activation='relu')(dense_1_1)
		dense_2 = Dropout(p)(dense_1)
		dense_2 = Dense(4096, activation='relu')(dense_2)
		dense_3 = Dropout(p)(dense_2)  
		dense_4 = Dense(1000)(dense_3)
		prediction = Activation("softmax",name="softmax")(dense_4)

		model = Model(input=inputs, output=prediction)

		if train:
			model.load_weights('weights/alexnet_weights.h5')

		for i in range(9):
			model.layers.pop()

		#up1 = UpSampling2D((15,15),dim_ordering='th')(conv_5_0)
		branches = []
		for i in range(21):
			branches.append(self.branchout(conv_5_0,'uniform'))

		prediction = merge(branches,mode='concat',concat_axis=1)
		#prediction = Activation('softmax')(prediction)
		prediction = Lambda(self.maxnorm,output_shape=self.maxnormshape)(prediction)

		model = Model(input=inputs, output=prediction)

		if not train:
			model.load_weights('newweights/alexvoc_branchavg.h5')

		self.model = model

		print(model.summary())
		self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['mse'])
		return self.model
	def get_layer_output(self,X,n):
		get_nth_layer_output = K.function([self.model.layers[0].input],[self.model.layers[n].output])
		layer_output = get_nth_layer_output([X])[0]
		return layer_output

	def grad_wrt_input(self,inputf): 
		fx = theano.function( [self.model.layers[0].input] ,T.jacobian(self.model.layers[-1].output.flatten(),self.model.layers[0].input), allow_input_downcast=True)
		grad = fx(inputf)
		return grad
