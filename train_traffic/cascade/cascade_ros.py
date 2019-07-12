#! /usr/bin/env python

import rospy
import cv2
from std_msgs.msg import String, Int8
from sensor_msgs.msg import CompressedImage
import numpy as np

def callback(data):
	np_arr = np.fromstring(data.data, np.uint8)
	arr =  cv2.imdecode(np_arr, cv2.IMREAD_COLOR)



if __name__ == '__main__':
	rospy.init_node('stop_sign')
	rospy.sleep(3)
	stop_cascade  = cv2.CascadeClassifier('/home/bowen/cascade_model/cascade.xml')
	print('Model loaded, begin to detect...')
	rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, callback, queue_size=1)
	rospy.Publisher('/stop_response', Int8)
	rospy.Rate(5)
	
