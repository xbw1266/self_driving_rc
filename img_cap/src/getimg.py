#! /usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import csv
import sys
import os
from std_msgs.msg import String



def save_img(data, name, current_dir, csv_name, angle):
	print('image saved!')
	filepath = current_dir + '/' + csv_name
	np_arr = np.fromstring(data.data, np.uint8)
	arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	cv2.imwrite(name, arr)
	with open(filepath, 'a') as f:
		writer = csv.writer(f)
		row = [str(rospy.get_time()), current_dir + '/' + name, angle]
		writer.writerow(row) 



def callback(data):

	# define some of the variables
	global i
	global t1
	global angle
	current_dir = os.getcwd()
	filename = 'img_loc.csv'

	if i == 0:
		angle = 90	
		save_img(data, 'img_0.jpeg', current_dir, filename, angle)
		print(angle)
		t1 = rospy.get_time()
		i += 1		
	elif rospy.get_time()-t1 >= 1:
		nam = 'img_{}.jpeg'.format(i)
		save_img(data, nam, current_dir, filename, angle)
		t1 = rospy.get_time()
		i += 1
	else:
		pass



def callback_key(data):
	delta = 3
	global angle
	if data.data == 'L':
		angle -= delta
	elif data.data == 'R':
	 	angle += delta



def main():
	rospy.init_node('image_listener')
	image_topic = '/raspicam_node/image/compressed'
	key_topic = '/keys'
	rospy.Subscriber(image_topic, CompressedImage, callback, queue_size=1)
	rospy.Subscriber(key_topic, String, callback_key, queue_size=1)
	rospy.spin()
	
if __name__ == "__main__":
	i = 0
	main()	

