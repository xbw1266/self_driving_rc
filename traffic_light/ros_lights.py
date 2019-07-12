import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Int8
import rospy


class Detect:
	def __init__(self, img_topic):
		self.sub = rospy.Subscriber(img_topic, CompressedImage, self.detect_lights, queue_size=1)
		self.pub = rospy.Publisher('/lights', String, queue_size=1)
		self.status = 'N'
		self.radius = 0
		self.green_low = np.array([50,20,10])
		self.green_high = np.array([95, 150, 255])
		self.red_high = np.array([200,120,255])
		self.red_low = np.array([130,10,100])
		rospy.Rate(5)


	def detect_lights(self, img_data):
		np_arr = np.fromstring(img_data.data, np.uint8)
		arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
		hsv = cv2.GaussianBlur(hsv, (5,5), 0)
		red_mask = cv2.inRange(hsv, self.red_low, self.red_high)
		green_mask = cv2.inRange(hsv, self.green_low, self.green_high)	
		v_green = self.image_process(hsv, green_mask)
		v_red = self.image_process(hsv, red_mask) 
		cnt_g, a_g = self.find_circles(v_green)	
		cnt_r, a_r = self.find_circles(v_red)
		if len(cnt_g) == 0 and len(cnt_r) == 0:
			self.status = 'N'
		elif len(cnt_g) == 0: # red light:
			if a_r > 500:
				print(a_r)
				self.status = 'R'
	#		else:
	#			self.status = 'N'
			arr = cv2.drawContours(arr, cnt_r, -1, (0,255,0),3)
			cv2.putText(arr, 'RED LIGHT', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1,cv2.LINE_AA)

		elif len(cnt_r) == 0:
			self.status = 'G'
			arr = cv2.drawContours(arr, cnt_g, -1, (0,255,0),3)
			cv2.putText(arr, 'GREEN LIGHT', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1,cv2.LINE_AA)


	#	if num_g == 0 and num_r == 0:
	#		self.status = 'N'
	#	elif num_g >= 1:
	#		self.status = 'G'
	#		for i  in cir_g[0,:]:
	#		     # draw the outer circle
	#			cv2.circle(arr,(i[0],i[1]),i[2],(0,255,0),2)
	#		     # draw the center of the circle
	#			cv2.circle(arr,(i[0],i[1]),2,(0,0,255),3)
	#			cv2.putText(arr, 'Green light', (i[0] - 2*i[2],i[1] - 2*i[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1,cv2.LINE_AA)
	#		self.radius = i[2]
	#	elif num_r >= 1:
	#		for i  in cir_r[0,:]:
	#		     # draw the outer circle
	#			cv2.circle(arr,(i[0],i[1]),i[2],(0,255,0),2)
	#		     # draw the center of the circle
	#			cv2.circle(arr,(i[0],i[1]),2,(0,0,255),3)
	#			cv2.putText(arr, 'Red light', (i[0] - 2*i[2],i[1] - 2*i[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1,cv2.LINE_AA)
	#			self.radius = i[2]
	#		if self.radius > 12:
	#			self.status = 'R'
	#		else:
	#			self.status = 'G'
		
		cv2.imshow('img', arr)
		cv2.imshow('v_red', v_red)
		cv2.imshow('v_green', v_green)
		k = cv2.waitKey(30)
		self.pub.publish(self.status)
		#print(self.status)
		if k == 27:
			cv2.destroyAllWindows()
			 


	def image_process(self, hsv_image, mask):
		img = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
		_, _, v = cv2.split(img)
		v = cv2.GaussianBlur(v, (5,5), 0)
		return v
	

	def find_circles(self, v_image):
		ret, th = cv2.threshold(v_image,127,255,0)
		contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cnt = contours
		big_cnt = []
		max_area = 0
		for i in cnt:
			area = cv2.contourArea(i)
			if(area > max_area):
				max_area = area
				big_cnt = i

		if max_area < 300 or max_area > 2000:
			big_cnt = []
		#print(max_area)
		return big_cnt, max_area
##		circles = cv2.HoughCircles(v_image, cv2.HOUGH_GRADIENT,1, 500, param1=25,param2=25,minRadius=10,maxRadius=30)
#		if circles is not None:
#			circles = np.uint16(np.around(circles))
#			num = np.size(circles, 0)
#		else:
#			num = 0
#			circles = 0
#		return num, circles		


if __name__ == '__main__':
	rospy.init_node('detect_lights')
	rospy.sleep(1)
	detect = Detect('/raspicam_node/image/compressed')
	rospy.spin()	
