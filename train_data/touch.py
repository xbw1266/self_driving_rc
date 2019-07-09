from keras.models import load_model
import cv2
import numpy as np
from data_preprocess import Img_process
from keras.optimizers import Adam


def predict(img):
	model = load_model('model_wasd.h5')
	img = Img_process(img)
	img_r = img.resize(320,160)
	img_h = img.hsv()
	img_b = img.blur()
	img_d = img.detect()
	img_d = np.expand_dims(img_d, axis=0)
	key_map = ['w', 'a', 'd'] 
	m = model.predict(img_d)
	key = key_map[np.argmx(m)]
	print(key)



path = '/home/bowen/data/'
image1 = cv2.imread(path + 'img_200.jpeg')
predict(image1)

