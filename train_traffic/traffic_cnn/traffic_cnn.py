from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten, Dropout
from keras.layers import Dense, Lambda
import cv2
import numpy as np
from keras.optimizers import Adam

class Traffic_CNN:
	def __init__(self, X_train, Y_train):
		self.X_train, self.Y_train = X_train, Y_train
		model = Sequential()
		model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(32,32,3)))
		model.add(Convolution2D(32,5,5, activation='relu'))
		model.add(Convolution2D(32,5,5, activation='relu'))
		model.add(Convolution2D(64,3,3, activation='relu'))
		model.add(Convolution2D(64,3,3, activation='relu'))
		model.add(Convolution2D(128,3,3, activation='relu'))
		model.add(Convolution2D(128,3,3, activation='relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Dense(43,activation='softmax'))
		self.model = model
		print('Model built')
		print(self.model.summary())

	def train(self):
		INIT_LR = 1e-3
		EPOCHS = 30
		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
		self.model.fit(self.X_train, self.Y_train, validation_split=0.2, shuffle=True, epochs=EPOCHS)
		self.model.save('traffic.h5', save_best_only=True)
		print('Model trained and saved as traffic.h5')




