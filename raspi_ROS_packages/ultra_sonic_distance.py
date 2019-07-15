#!/usr/bin/env python

import rospy
import time
import RPi.GPIO as GPIO
from std_msgs.msg import Float64

GPIO.setwarnings(False)

def get_distance():

    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    start = time.time()
    #start = rospy.get_rostime()

    while GPIO.input(GPIO_ECHO)==0:
        start = time.time()
	#start = .get_rostime()
    while GPIO.input(GPIO_ECHO)==1:
        stop = time.time()
    #stop = rospy.get_rostime()
    elapsed = stop-start
    distance = (elapsed * 34300)/2

    return distance


def initial():
	pub = rospy.Publisher('front_distance', Float64, queue_size=10)
	rospy.init_node('distance_sensor')
	rate = rospy.Rate(10)
	rospy.loginfo('node start')
#	rospy.spin()
	while not rospy.is_shutdown():
		frontDistance = get_distance()
		rospy.loginfo(frontDistance)
		pub.publish(frontDistance)
		rate.sleep()
	rospy.spin()
# referring to the pins by GPIO numbers
GPIO.setmode(GPIO.BOARD)

# define pi GPIO
GPIO_TRIGGER = 19
GPIO_ECHO    = 21

# output pin: Trigger
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)
# input pin: Echo
GPIO.setup(GPIO_ECHO,GPIO.IN)
# initialize trigger pin to low
GPIO.output(GPIO_TRIGGER, False)

if __name__ == '__main__':
#	try:
	initial()

#	except rospy.ROSInterruptException:
#		GPIO.cleanup()
#		pass
