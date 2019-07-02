import csv
import cv2
import numpy as np


filepath = '/home/bowen/data/'
filename = 'record.csv'
lines = []
with open(filepath + filename) as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)


images = []
measurements = []
for line in lines:
	img_path = line[1]
	image = cv2.imread(img_path)
	images.append(image)
	measurement = float(line[-1])
	measurements.append(measurement)
	
	# now adding data agumentation:
	images.append(cv2.flip(image))
	measurements.append(measurement * -1.0)


print('image loaded, the dimension is {}'.format(image.shape))

if image.shape != (160.320,3):
	print('The dimension is not comptiable, resizing ...')
	for idx, image in enumerate(images):
		new_img = cv2.resize(image, (320,160))
		images[idx] = new_img
 
X_train = np.array(images)
Y_train = np.array(measurements)


# the following is the structures of the CNN:
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)) # normalize the data and centralize the data
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

	
