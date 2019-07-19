import cv2
import numpy as np
import glob

stop_cascade = cv2.CascadeClassifier('/home/bowen/cascade_model/cascade.xml')
cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()
#data_dir = '/home/bowen/14'
#for i in glob.glob(data_dir + '/*.png'):
	#img = cv2.imread(i)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	stops = stop_cascade.detectMultiScale(gray)
	for (x,y,w,h) in stops:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
		cv2.putText(img, 'Stop sign', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)	
	cv2.imshow('img', img)
	k = cv2.waitKey(1)
	if k == 27:
		break
		cv2.destroyAllWindows()

cap.release()



