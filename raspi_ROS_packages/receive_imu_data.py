#!/usr/bin/env python 

import rospy
import serial
from std_msgs.msg import String, Float64 
import math
from sensor_msgs.msg import Imu

port = serial.Serial("/dev/ttyS0", baudrate = 115200, timeout =0.1)

imu_raw = Imu()

def get_imu():
        rospy.init_node('receive_imu')
        pub = rospy.Publisher("arduino_imu/raw",Imu, queue_size=1)
	rate = rospy.Rate(50)
	seq = 0
	while not rospy.is_shutdown():
        	## get imu data from serial buffer
		rcvdStr = port.read_until()
		sax,say,saz,swx,swy,swz = rcvdStr.split(':')
		#print(rcvdStr)
		ax = float(sax) * 9.81
		ay = float(say) * 9.81
		az = float(saz) * 9.81
		wx = float(swx) * math.pi / 180
                wy = float(swy) * math.pi / 180
                wz = float(swz) * math.pi / 180 
#		print(ax, ay, az, wx, wy, wz)
		#print(ax*2.5)
		imu_raw.header.stamp = rospy.Time.now()
		imu_raw.header.frame_id = 'imu_link'
		imu_raw.header.seq = seq
		imu_raw.orientation_covariance[0] = -1
		imu_raw.linear_acceleration.x = ax
                imu_raw.linear_acceleration.y = ay
                imu_raw.linear_acceleration.z = az
		imu_raw.linear_acceleration_covariance[0] = -1
		imu_raw.angular_velocity.x = wx
                imu_raw.angular_velocity.y = wy
                imu_raw.angular_velocity.z = wz
                imu_raw.angular_velocity_covariance[0] = -1
		pub.publish(imu_raw)

		seq +=1
		rate.sleep()

if __name__=='__main__':

	get_imu()















