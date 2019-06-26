#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import csv
import sys
import time


opt = sys.argv[-1]
if opt == 'new':
	write_opt = 'w'
else:
	opt == 'append'
	write_opt = 'a'
ss = "/home/bowen/Desktop/key_rec.csv"


def callback(data):
	key_ = data.data
	rospy.loginfo(rospy.get_caller_id() + "I heard %s", key_)
	with open(ss, write_opt) as f:
		writer = csv.writer(f)
		row = [str(time.time()), key_]
		writer.writerow(row)


def listener():
	rospy.init_node('listener', anonymous=True)
	rospy.Subscriber("keys", String, callback)
	
	rospy.spin()

if __name__ == '__main__':
	print(opt)
	listener()

