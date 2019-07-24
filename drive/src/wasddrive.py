#! /usr/bin/env python

import time
from keras.models import load_model
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import numpy as np
from data_preprocess import Img_process
import cv2
<<<<<<< HEAD
from robot_decisions.msg import behavior_msg
=======
>>>>>>> master
print('Packages loaded')

class Drive:
	def __init__(self, model, img_topic):
		self.model = load_model(model)
		self.model._make_predict_function()
		self.sub1 = rospy.Subscriber(img_topic, CompressedImage, self.img_process)
		self.sub2 = rospy.Subscriber(img_topic, CompressedImage, self.stop_process)
<<<<<<< HEAD
	#	self.sub3 = rospy.Subscriber('/lights', String, self.dt_lights)
		self.pub = rospy.Publisher('/keys', String, queue_size=1)
		self.pubstop = rospy.Publisher('behavior_msg',behavior_msg,queue_size=1)
=======
		self.sub3 = rospy.Subscriber('/lights', String, self.dt_lights)
		self.pub = rospy.Publisher('/keys', String, queue_size=1)
>>>>>>> master
		self.cascade = cv2.CascadeClassifier('/home/bowen/cascade_model/cascade.xml')
		self.stop = False
		self.stop_action = False
		self.x = 100
		self.w = 100
		self.t1 = 0
		rospy.Rate(5)
		self.id = 1
		self.stop_light = False
		self.t2 = 0
<<<<<<< HEAD
	
		##behavior msg init
		self.behavior = behavior_msg() 
		self.behavior.stop_sign = False
		self.behavior.traffic_light = False
=======
>>>>>>> master
		
	def img_process(self, img_data):
		np_arr = np.fromstring(img_data.data, np.uint8)
		arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		self.image = arr
		image = Img_process(arr)
		r = image.resize(320,160)
		h = image.hsv()
		b = image.blur()
		image_d = image.detect()
		self.img = np.expand_dims(image_d, axis=0)
		self.predict()
		#print(self.x, self.w)
		if (self.stop_light):
			self.pub.publish(' ')
		else:
			if (self.stop) and (self.w > 50 and (self.x < 20 or self.x > 470)) and (time.time() - self.t2 > 5.0):
				if self.id == 1:
					print('Timer triggered!')
					self.t1 = time.time()
					self.stop_action = True
					self.id += 1
			if (time.time() - self.t1 >= 4.8) and self.stop_action is True:
				self.pub.publish(' ')
				rospy.sleep(5)
				self.stop_action = False
				self.t2 = time.time()
				self.id = 1
			self.pub.publish(self.key)
		#print('Sending command: {}'.format(self.key))
		cv2.imshow('converted', image_d)
		k = cv2.waitKey(1)
		if k == 27:
			cv2.destroyAllWindows()
<<<<<<< HEAD
		self.behavior.move = self.key
		
=======
>>>>>>> master

	def predict(self):
		key_map = ['w', 'a', 'd']
		self.idx = self.model.predict(self.img)
		self.key = key_map[np.argmax(self.idx)]
		

	def stop_process(self, data):
		np_arr = np.fromstring(data.data, np.uint8)
		arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	#	while self.image is not None:
	#	gray = cv2.resize(self.image, (320,160))
		gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
<<<<<<< HEAD
		if len(stops) <= 0:
			stops = self.cascade.detectMultiScale(gray)
=======
		stops = self.cascade.detectMultiScale(gray)
		if len(stops) <= 0:
>>>>>>> master
			self.stop = False
			cv2.imshow('stop_sign', gray)
			k = cv2.waitKey(1)
			if k == 27:
				cv2.destroyAllWindows()
		else:
			self.stop = True
			for (x,y,w,h) in stops:
				self.w = w
				self.x = x
				#print('x: {}, w: {}'.format(x, w))
				cv2.rectangle(gray, (x,y), (x+w, y+h), (255,0,0), 2)
				cv2.putText(gray, 'Stop sign', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)				
			cv2.imshow('stop_sign', gray)
			k = cv2.waitKey(1)
			if k == 27:
				cv2.destroyAllWindows()
<<<<<<< HEAD
		self.behavior.stop = self.stop
=======
		
>>>>>>> master
		
	def dt_lights(self,data):
		print(data.data)
		if data.data == 'R':
			print('Stop!!!!!')
			rospy.sleep(1)
			self.stop_light = True
			self.pub.publish(' ')
		elif data.data == 'G':
			self.stop_light = False
<<<<<<< HEAD
		self.behavior.traffic_light = self.stop_light
=======

>>>>>>> master



		
if __name__ == "__main__":
	rospy.init_node('drive')
	rospy.sleep(3)
	print('Waiting 3 seconds to start')
	drive = Drive('model_wasd.h5', '/raspicam_node/image/compressed')
	rospy.spin()
