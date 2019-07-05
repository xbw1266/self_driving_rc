from keras.models import load_model
import cv2
import numpy as np
from data_preprocess import Img_process


def predict(img):
	model = load_model('model.h5')
	model.compile(loss='mse', optimizer='adam')
	img = Img_process(img)
	img_r = img.resize(320,160)
	img_h = img.hsv()
	img_b = img.blur()
	img_d = img.detect()
	img_d = np.expand_dims(img_d, axis=0)
	y1, y2 = np.array(model.predict(img_d), dtype=np.uint8)
	print(y1, y2)


path = '/home/bowen/data/'
image1 = cv2.imread(path + 'img_200.jpeg')
predict(image1)

