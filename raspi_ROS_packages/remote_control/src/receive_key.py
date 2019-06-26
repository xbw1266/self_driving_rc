#!/usr/bin/env python

import rospy
import serial
from std_msgs.msg import String

port = serial.Serial("/dev/ttyS0", baudrate = 9600, timeout =0.1)

def callback(data):
	#print(data.data)
	port.write(data.data)		

def receive_key():
	rospy.init_node('receive_key',anonymous = True)
	rospy.Subscriber("keys",String,callback)	
	rospy.spin()

if __name__=='__main__':
	receive_key()	
