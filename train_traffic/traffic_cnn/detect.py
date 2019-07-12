import cv2
from keras.models import load_model
import numpy as np
import os
import glob
import time

_dir = '/home/bowen/Downloads/Train/14/'
path = glob.glob(_dir + '*.png')
model = load_model('traffic.h5')
d_ts = []
for i in path:
	t1 = time.time()
	img = cv2.imread(i)
	img = cv2.resize(img, (25,25))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = np.expand_dims(img, axis=0)
	img = np.expand_dims(img, axis=0)
	m = model.predict(img)
	if np.argmax(m) == 14:
		print('Stop sign!')
	else:
		print('Not stop sign')
	d_t = time.time() - t1
	d_ts.append(d_t)

print('The average fps: {}'.format((len(d_ts)/sum(d_ts))))
