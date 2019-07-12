import cv2
import numpy as np


def detect_lights(img):
	img = cv2.imread(img)
	img = cv2.resize(img, (img.shape[1]//1, img.shape[0]//1))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_blur = cv2.GaussianBlur(new_img, (5,5), 0)
#	n1 = 90
#	n2 = 0
#	n3 = 0
#while n1 <= 91:
	#n1 += 1
	#print(n1)
	green_low = np.array([50, 0, 200])
	green_high = np.array([91, 200, 255])
	green_mask = cv2.inRange(img_blur, green_low, green_high)
	green = cv2.bitwise_and(img_blur, img_blur, mask=green_mask)
	
	red_low = np.array([100,40,200])
	red_high = np.array([255,80,255])
	red_mask = cv2.inRange(img_blur, red_low, red_high)
	red = cv2.bitwise_and(img_blur, img_blur, mask=red_mask)
	cv2.imshow('red', red)
	k = cv2.waitKey(0)
	h,s,v = cv2.split(green)
	v = cv2.GaussianBlur(v, (5,5), 0)
#	cv2.imshow('v', v)
#	k = cv2.waitKey(0)
	circles = cv2.HoughCircles(v, cv2.HOUGH_GRADIENT,1, 1, param1=100,param2=30,minRadius=10,maxRadius=0)
	#print(circles)
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
	    # draw the outer circle
	    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
	    # draw the center of the circle
	    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
	cv2.putText(img, 'Green light', (i[0] - 2*i[2],i[1] - 2*i[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1  ,cv2.LINE_AA)
	cv2.imshow('img', img)
	k = cv2.waitKey(0)
	if k == 27:
		cv2.destroyAllWindows()


if __name__  == '__main__':	
	detect_lights('/home/bowen/Desktop/img_2.jpeg')	
