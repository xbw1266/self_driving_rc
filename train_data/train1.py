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
key_map = {"a": [15, 30], "w": [25, 25], "s": [-25,-25], "d": [30, 15], " ": [0, 0]}

with open(filepath + filename) as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)


images = []
measurements_Y1 = []
measurements_Y2 = []
for idx, line in enumerate(lines):
	img_path = line[1]
	image = cv2.imread(img_path)
	images.append(image)
	measurement_Y1 = key_map[line[-1]][0]
	measurement_Y2 = key_map[line[-1]][1]
	measurements_Y1.append(measurement_Y1)
	measurements_Y2.append(measurement_Y2)
	
	image_extra = cv2.flip(image, 1)
	if idx == 1:		
#		cv2.imshow('img_after', image_extra)
#		cv2.imshow('img_before', image)
		k = cv2.waitKey(0)
		if k == 27:
			cv2.destroyAllWindows()
	
	images.append(image_extra)
	if line[-1] == 'w':	
		measurements_Y1.append(key_map[line[-1]][0])
		measurements_Y2.append(key_map[line[-1]][1])
	elif line[-1] == 'a':
		new_key = 'd'
		measurements_Y1.append(key_map[new_key][0])
		measurements_Y2.append(key_map[new_key][1])
	elif line[-1] == 'd':
		new_key = 'a'	
		measurements_Y1.append(key_map[new_key][0])
		measurements_Y2.append(key_map[new_key][1])
	elif line[-1] == ' ':
		measurements_Y1.append(0)
		measurements_Y2.append(0)
#	# now adding data agumentation:
#	images.append(cv2.flip(image, 1))
#	measurements.append(180-measurement)

print(images[0].shape)

from data_preprocess import Img_process
from matplotlib import pyplot as plt

print('image loaded, the dimension is {}'.format(image.shape))
if image.shape != (160,320,3):
	print('The dimension is not comptiable, resizing ...')
	for idx, image in enumerate(images):
		image = Img_process(image)
		image_resized = image.resize(320, 160)
		image_gray = image.hsv()
		image_blur = image.blur()
		image_edge = image.detect()
		image_binary = image.binary(50)
		if idx == 20:
			plt.subplot(221), plt.imshow(image_gray, 'hsv'), plt.title('hsv image')		
			plt.subplot(222), plt.imshow(image_blur, 'hsv'), plt.title('blurred image')		
			plt.subplot(223), plt.imshow(image_edge, 'hsv'), plt.title('hsv edge image')		
			plt.subplot(224), plt.imshow(image_binary[:,:,0])
			plt.show()
			#print(image_binary)
		
		images[idx] = image_edge
		#images[idx] = image_binary
X_train = np.array(images)
Y_train1 = np.array(measurements_Y1)
Y_train2 = np.array(measurements_Y2)
#from RCNN import Mycnn
print(X_train.shape, Y_train1.shape, Y_train2.shape)
#my_model = Mycnn(X_train, Y_train1, Y_train2)
#my_model.train()
#
## the following is the structures of the CNN:
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#
#
model = Sequential()
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

	
