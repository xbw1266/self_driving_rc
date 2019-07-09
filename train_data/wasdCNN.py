from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Lambda
import cv2
import numpy as np
from keras.optimizers import Adam

class Mycnn:
	def __init__(self, X_train, Y_train):
		self.X_train, self.Y_train = X_train, Y_train
		model = Sequential()
		model.add(Lambda(lambda x: x/ 255.0 -0.5, input_shape=(160,320,3)))
		model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
		model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
		model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
		model.add(Convolution2D(64,3,3, activation='relu'))
		model.add(Convolution2D(64,3,3, activation='relu'))
		model.add(Flatten())
		model.add(Dense(100))
		model.add(Dense(50))
		model.add(Dense(10))
		model.add(Dense(3,activation='softmax'))
		self.model = model
		print('Model built')
		print(self.model.summary())

	def train(self):
		INIT_LR = 1e-3
		EPOCHS = 3
		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
		self.model.fit(self.X_train, self.Y_train, validation_split=0.2, shuffle=True, epochs=EPOCHS)
		self.model.save('model_wasd.h5')
		print('Model trained and saved as model.h5')



