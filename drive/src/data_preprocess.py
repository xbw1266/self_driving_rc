import numpy as np
import cv2



class Img_process:
	def __init__(self, img):
		self.img = img
		w, h, c = img.shape
		self.w = w
		self.h = h


	def resize(self, w, h):
		self.w = w
		self.h = h
		self.img = cv2.resize(self.img, (w, h))
		return self.img

	def hsv(self):
		new_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
	#	new_img = np.array(new_img)
	#	new_img = new_img.reshape(self.w, self.h, 1)
		self.hsv = new_img
		return self.hsv

	def blur(self):		
		self.blur = cv2.GaussianBlur(self.hsv, (5, 5), 0) 
		return self.blur


	def detect(self):
		mask  = cv2.inRange(self.blur, (20,50,100), (35,255,255))
		target = cv2.bitwise_and(self.blur, self.blur, mask = mask)
		self.target = target
		return self.target

		
