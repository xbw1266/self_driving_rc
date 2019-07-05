#! /usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
import csv
import sys
import os
from std_msgs.msg import String
import numpy as np


class Process:
	def __init__(self, img_topic, key_topic, delta=3):
		self.img_topic, self.key_topic = img_topic, key_topic		
		self.img_sub = rospy.Subscriber(self.img_topic, CompressedImage, self.img_callback, queue_size=1)
		self.key_sub = rospy.Subscriber(self.key_topic, String, self.key_callback, queue_size=1)
		#self.delta = delta
		self.i = 0
		self.filename = 'record.csv'
		#self.angle = 90
		self.dir = os.getcwd()
		self.t1 = 0
		self.key = ' '



	def img_callback(self, img_data):
		if self.i == 0:
			self.save_process(img_data, 'img_0.jpeg', self.dir, self.filename, self.key)
			self.t1 = rospy.get_time()	
			self.i += 1
		elif rospy.get_time() - self.t1 >= 0.2:
			name = 'img_{}.jpeg'.format(self.i)
			self.save_process(img_data, name, self.dir, self.filename, self.key)
			self.t1 = rospy.get_time()
			self.i += 1



	def key_callback(self, key_data):
#		if key_data.data == 'a':
#			self.angle -= self.delta
#		elif key_data.data == 'd':
#			self.angle += self.delta
		self.key = key_data.data	


	def save_process(self, msg, name, current_dir, csv_name, key):
		print('{} image saved'.format(self.i))
		np_arr = np.fromstring(msg.data, np.uint8)
		arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		cv2.imwrite(name, arr)		
		filepath = current_dir + '/' + csv_name
		with open(filepath, 'a') as f:
			writer = csv.writer(f)
			row = [str(rospy.get_time()), current_dir + '/' +  name, key]
			writer.writerow(row)
	
		

if __name__ == '__main__':
	rospy.init_node('image_process')
	rospy.sleep(2)
	my_process = Process('/raspicam_node/image/compressed', '/keys')
	rospy.spin()

