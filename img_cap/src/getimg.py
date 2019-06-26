#! /usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()


def callback(data):
	print('received image!')
	try:
		cv2_img = bridge.imgmsg_to_cv2(data, 'bgr8')
	except CvBridgeError, e:
		print(e)

	else:
		global i
		nam = 'img_{}.jpg'.format(i)
		cv2.imwrite(nam, cv2_img)
		rospy.sleep(1)	
		i += 1 

def main():
	rospy.init_node('image_listener')
	image_topic = "cv_camera/image_raw"
	rospy.Subscriber(image_topic, Image, callback)
#	rospy.sleep(1)
	rospy.spin()

	
if __name__ == "__main__":
	i = 0	
	main()	
