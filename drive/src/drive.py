#! /usr/bin/env python

from keras.models import load_model
import cv2
import rospy
from snesor_msgs.msg import CompressedImage
from std_msgs.msg import UInt8MultiArray
import numpy as np
from data_preprocess import Img_process


class Drive:
	def __init__(self, model, img_topic):
		self.model = load_model(model)
		self.sub = rospy.Subscriber(img_topic, CompressedImage, self.img_process, queue_size=1)
		self.pub = rospy.Publisher('/duty_cycle', UInt8MultiArray, queue_size=1)
		
	def img_process(self, img_data):
		np_arr = np.fromstring(img.data, np.uint8)
		arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		imgage = Img_process(arr)
		r = image.resize(320,160)
		h = image.hsv()
		b = image.blur()
		image_d = image.detect()
		image_d = np.expand_dims(img_d, axis=0)
		self.img = image_d
		self.predict()
		self.pub.publish(self.action)
		return self.img 


	def predict(self):
		self.y1, self.y2 = np.array(self.model.predict(self.img), dtype=np.uint8)
		self.action = UInt8MultiArray(data=[self.y1, self.y2])	
		

if __name__ == "main":
	rospy.init_node('drive')
	rospy.sleep(3)
	drive = Drive('model.h5', '/raspicam_node/image/compressed')
	rospy.spin()
