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

def oquabloss(y_true,y_pred):
	return K.sum(K.log(1.0+K.exp(-1.0*y_true*y_pred)))

def maxnorm(vector):
	return 1.0*vector/K.max(vector)
	
def maxnormshape(shape):
	return shape

class Architectures:
	def vgg19(self):
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
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
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
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(Flatten())
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1000, activation='softmax'))

		model.load_weights('weights/vgg19_weights.h5')

		for layer in model.layers:
			layer.trainable = False
		for i in range(2):
			model.layers.pop()

		model.add(Dense(20,activation='sigmoid',init='glorot_uniform'))

		model.compile(loss='categorical_crossentropy',optimizer='sgd')
		print(model.summary())

		self.model = model
		return model

	
	def vgg16(self):
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
		model.add(Dropout(0.5))
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1000, activation='softmax'))


		model.load_weights('weights/vgg16_weights.h5')

		'''
		for layer in model.layers:
			layer.trainable = False
		'''

		for i in range(2):
			model.layers.pop()

		model.outputs = [model.layers[-1].output]
		model.layers[-1].outbound_nodes = []

		model.add(Dense(20,activation='relu',init='glorot_uniform'))

		print(model.summary())
	
		model.compile(loss='categorical_crossentropy',optimizer='sgd')

		return model

	def alexnet(self):
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
		dense_2 = Dropout(0.5)(dense_1)
		dense_2 = Dense(4096, activation='relu')(dense_2)
		dense_3 = Dropout(0.5)(dense_2)  
		dense_4 = Dense(1000)(dense_3)
		prediction = Activation("softmax",name="softmax")(dense_4)

		model = Model(input=inputs, output=prediction)

		model.load_weights('weights/alexnet_weights.h5')

		for i in range(3):
			model.layers.pop()

		for layer in model.layers:
			layer.trainable = False

		'''
		dense_1 = Dense(100, activation='tanh',init='he_normal')(dense_1_1)
		dense_1 = BatchNormalization()(dense_1)
		#dense_2 = Dropout(0.5)(dense_1)
		dense_2 = Dense(100, activation='tanh',init='he_normal')(dense_1)
		dense_2 = BatchNormalization()(dense_2)
		#dense_3 = Dropout(0.5)(dense_2)  
		dense_3 = Dense(100,init='he_normal')(dense_2)
		'''

		dense_4 = Dense(20,init='glorot_uniform')(dense_2)

		prediction = Activation("sigmoid",name="softmax")(dense_4)

		model = Model(input=inputs, output=prediction)

		self.model = model

		#optimizers
		sgd = SGD(lr=1e-4)
		adam = Adam(lr=1e-3)
		self.model.compile(loss='mse',optimizer=sgd,metrics=['mse'])
		return self.model

	def get_layer_output(self,X,n):
		get_nth_layer_output = K.function([self.model.layers[0].input],[self.model.layers[n].output])
		layer_output = get_nth_layer_output([X])[0]
		return layer_output
