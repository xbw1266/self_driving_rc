import csv
import numpy as np
#import sys
#
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#print(sys.path)


import cv2

filepath = '/home/linjian/Downloads/Raw data/data/'
filename = 'record.csv'
lines = []
#key_map = {"a": [15, 30], "w": [25, 25], "s": [-25,-25], "d": [30, 15], " ": [0, 0]}

key_map = {"a": 0, "w": 1, "s": 3, "d": 2}
with open(filepath + filename) as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)

lines.pop(0) #remove the first line, which has empty imput

images = []
measurements_Y1 = []
measurements_Y2 = []
for idx, line in enumerate(lines):
	img_path = line[1]
	image = cv2.imread(img_path)
	images.append(image)
#	measurement_Y1 = to_categorical(key_map[line[-1]])
	measurement_Y1 = key_map[line[-1]]
	measurements_Y1.append(measurement_Y1)
	
	image_extra = cv2.flip(image, 1)
	if idx == 1:		
#		cv2.imshow('img_after', image_extra)
#		cv2.imshow('img_before', image)
		k = cv2.waitKey(0)
		if k == 27:
			cv2.destroyAllWindows()
	
	images.append(image_extra)
	if line[-1] == 'w':	
		#measurements_Y1.append(to_categorical(key_map[line[-1]]))
		measurements_Y1.append(key_map[line[-1]])
	elif line[-1] == 'a':
		new_key = 'd'
		#measurements_Y1.append(to_categorical(key_map[new_key]))
		measurements_Y1.append(key_map[new_key])
	elif line[-1] == 'd':
		new_key = 'a'	
		#measurements_Y1.append(to_categorical(key_map[new_key]))
		measurements_Y1.append(key_map[new_key])
#	# now adding data agumentation:
#	images.append(cv2.flip(image, 1))
#	measurements.append(180-measurement)

print(images[0].shape)

from data_preprocess import Img_process
from matplotlib import pyplot as plt
IMAGE_SIZE = 160
print('image loaded, the dimension is {}'.format(image.shape))
if image.shape != (IMAGE_SIZE,IMAGE_SIZE,3):
	print('The dimension is not comptiable, resizing ...')
	for idx, image in enumerate(images):
		image = Img_process(image)
		image_resized = image.resize(IMAGE_SIZE, IMAGE_SIZE)
		image_gray = image.hsv()
		image_blur = image.blur()
		image_edge = image.detect()
		image_binary = image.binary(50)
		if idx == 20:
			plt.subplot(221), plt.imshow(image_gray, 'hsv'), plt.title('hsv image')		
			plt.subplot(222), plt.imshow(image_blur, 'hsv'), plt.title('blurred image')		
			plt.subplot(223), plt.imshow(image_edge, 'hsv'), plt.title('hsv edge image')		
			plt.subplot(224), plt.imshow(image_binary[:,:,0]), plt.title('binary image')
			plt.show()
		
		images[idx] = image_edge #use edge image as module input
		#images[idx] = image_binary #use binary image as module input

from keras.utils import to_categorical

X_train = np.array(images)
Y_train1 = np.array(to_categorical(measurements_Y1))
#from RCNN import Mycnn
print(X_train.shape, Y_train1.shape)
print(Y_train1[0],Y_train1[20])
#my_model = Mycnn(X_train, Y_train1, Y_train2)
#my_model.train()
#
## the following is the structures of the CNN:
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#
#create model
model = Sequential()
#model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu',input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(Convolution2D(32,3,3,subsample=(2,2),activation='relu',input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
#model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
#model.add(Dense(50))
model.add(Dense(3,activation='softmax'))

#compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train, Y_train1, validation_split=0.2, shuffle=True, nb_epoch=3)

#add model layers
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)) # normalize the data and centralize the data
#model.add(Convolution2D(6,5,5,activation='relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation='relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))
#
#model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
#
#model.save('model.h5')
#

	
