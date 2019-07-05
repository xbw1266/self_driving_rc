from keras.models import Model
from keras.layers import Flatten, Dense, Lambda, Input, Embedding
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import cv2
import numpy as np


class Mycnn:
	def __init__(self, X_train, Y_train1, Y_train2):
		self.X_train, self.Y_train1 = X_train, Y_train1
		self.Y_train2 = Y_train2
		self.X = Input(shape=(160,320,3))
		#x = Lambda(lambda x: x / 255.0 - 0.5)(self.X)
		x = Convolution2D(24,5,5,subsample=(2,2), activation='relu', input_shape=(160,320,3))(self.X)
		x = Convolution2D(36,5,5,subsample=(2,2),activation='relu')(x)
		x = Convolution2D(48,5,5,subsample=(2,2),activation='relu')(x)
		x = Convolution2D(64,3,3,activation='relu')(x)
		x = MaxPooling2D()(x)
		x = Convolution2D(64,3,3,activation='relu')(x)
		x = Flatten()(x)
		x = Dense(100)(x)
		x = Dense(50)(x)
		x = Dense(10)(x)
		self.Y1 = Dense(1)(x)
		self.Y2 = Dense(1)(x)
		self.model = Model(inputs=self.X, outputs=[self.Y1, self.Y2])
		print('Model built')
		print(self.model.summary())

	def train(self):
		self.model.compile(loss='mse', optimizer='adam')
		self.model.fit(self.X_train, [self.Y_train1, self.Y_train2], validation_split=0.2, shuffle=True, epochs=10)
		self.model.save('model.h5')
		print('Model trained and saved as model.h5')



