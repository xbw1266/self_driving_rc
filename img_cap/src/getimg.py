#! /usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import csv
import sys
import os
import glob



def save_img(data, name, current_dir, csv_name):
	print('image saved!')
	filepath = current_dir + '/' + csv_name
	np_arr = np.fromstring(data.data, np.uint8)
	arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	cv2.imwrite(name, arr)
	with open(filepath, 'a') as f:
		writer = csv.writer(f)
		row = [str(rospy.get_time()), current_dir + '/' + name]
		writer.writerow(row) 



def callback(data):

	# define some of the variables
	global i
	global t1
	current_dir = os.getcwd()
	filename = 'img_loc.csv'

	if i == 0:	
		save_img(data, 'img_0.jpeg', current_dir, filename)
		t1 = rospy.get_time()
		i += 1		
	elif rospy.get_time()-t1 >= 1:
		nam = 'img_{}.jpeg'.format(i)
		save_img(data, nam, current_dir, filename)
		t1 = rospy.get_time()
		i += 1
	else:
		pass

#	try:
#		print('image received!')
#		cv2_img
#		cv2_img = bridge.imgmsg_to_cv2(data, 'bgr8')
#	except CvBridgeError, e:
#		print(e)
#
#	else:
#		global i
#		nam = 'img_{}.jpg'.format(i)
#		cv2.imwrite(nam, cv2_img)
#		rospy.sleep(1)	
#		i += 1 

def main():
	rospy.init_node('image_listener')
	image_topic = '/raspicam_node/image/compressed'
	rospy.Subscriber(image_topic, CompressedImage, callback, queue_size=1)
	print('aaaaaaaaaa')
	rospy.spin()
	
if __name__ == "__main__":
	i = 0
	main()	

