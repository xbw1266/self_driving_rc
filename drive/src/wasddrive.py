#! /usr/bin/env python

from keras.models import load_model
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import numpy as np
from data_preprocess import Img_process
import cv2
print('Packages loaded')

class Drive:
	def __init__(self, model, img_topic):
		self.model = load_model(model)
		self.model._make_predict_function()
		self.sub = rospy.Subscriber(img_topic, CompressedImage, self.img_process, queue_size=1)
		self.pub = rospy.Publisher('/keys', String, queue_size=1)
		rospy.Rate(5)
		
	def img_process(self, img_data):
		np_arr = np.fromstring(img_data.data, np.uint8)
		arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		image = Img_process(arr)
		r = image.resize(320,160)
		h = image.hsv()
		b = image.blur()
		image_d = image.detect()
		self.img = np.expand_dims(image_d, axis=0)
		self.predict()
		self.pub.publish(self.key)
		print('Sending command: {}'.format(self.key))
		cv2.imshow('converted', image_d)
		k = cv2.waitKey(1)
		if k == 27:
			cv2.destroyAllWindows()



	def predict(self):
		key_map = ['w', 'a', 'd']
		self.idx = self.model.predict(self.img)
		self.key = key_map[np.argmax(self.idx)]
		

			
if __name__ == "__main__":
	rospy.init_node('drive')
	rospy.sleep(3)
	print('Waiting 3 seconds to start')
	drive = Drive('model_wasd.h5', '/raspicam_node/image/compressed')
	rospy.spin()
