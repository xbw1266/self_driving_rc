import cv2
import csv
import numpy as np
import pandas as pds
from keras.utils import to_categorical
import sys
import matplotlib.pyplot as plt


data_dir = '/home/bowen/Downloads/'
class_id_meta = []
images_meta = []

if sys.argv[-1] == 'True':
	print('Now preparing the class_id table...')
	with open(data_dir+'Meta.csv') as f:
		reader = csv.reader(f)
		for idx, line in enumerate(reader):
			if idx == 0:
				continue
			class_id_meta.append(line[1])
			image = cv2.imread(data_dir + line[0])
			images_meta.append(image)
	fig, axs =  plt.subplots(6,8, figsize=(32,32))
	axs = axs.ravel()
	for i in range(42):
		axs[i].imshow(cv2.cvtColor(images_meta[i], cv2.COLOR_BGR2RGB))
		axs[i].set_title('Class_{}'.format(class_id_meta[i]))
		axs[i].axis('off')
	plt.show()


class_id = []
images = []

with open(data_dir + 'Train.csv') as f:
	reader = csv.reader(f)
	for idx, line in enumerate(reader):
		if idx == 0:
			continue
		class_id.append(line[-2])
		path = line[-1]
		image = cv2.imread(data_dir + path)
		image = cv2.resize(image, (32,32))
	#	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#	image = np.expand_dims(image, axis=0) 
		images.append(image)			
 
print('Image processed successfully...')
print('{} images loaded'.format(len(images)))
print(image.shape)
X_train = np.array(images)
Y_train = to_categorical(class_id, num_classes=43)

from traffic_cnn import Traffic_CNN
model = Traffic_CNN(X_train, Y_train)
model.train()
