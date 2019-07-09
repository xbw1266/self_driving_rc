import csv
import cv2
import numpy as np
from keras.utils import to_categorical

filepath = '/home/bowen/data_new/'
filename = 'record.csv'
lines = []

with open(filepath + filename) as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)


key_num = {'w':0, 'a':1, 'd':2}
# w:0, a:1, d:2
images = []
measurements_Y = []
for line in lines:
	img_path = line[1]
	image = cv2.imread(img_path)
	images.append(image)
	measurement = line[-1]
	if measurement == ' ':
		measurement = 'w'
	measurements_Y.append(key_num[measurement])
	image_extra = cv2.flip(image, 1)
	images.append(image_extra)
	if measurement == 'w':
		new_m = 'w'
	elif measurement == 'a':
		new_m = 'd'
	elif measurement == 'd':
		new_m = 'a'
	measurements_Y.append(key_num[new_m])

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
		if idx == 1:
			plt.subplot(131), plt.imshow(image_gray, 'hsv'), plt.title('hsv image')		
			plt.subplot(132), plt.imshow(image_blur, 'hsv'), plt.title('blurred image')		
			plt.subplot(133), plt.imshow(image_edge, 'hsv'), plt.title('hsv edge image')		
			plt.show()
		images[idx] = image_edge

X_train = np.array(images)
Y_train = to_categorical(measurements_Y, num_classes=3)

from wasdCNN import Mycnn
my_model = Mycnn(X_train, Y_train)
my_model.train()

