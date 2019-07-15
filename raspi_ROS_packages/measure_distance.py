#!/usr/bin/env python

import time
import RPi.GPIO as GPIO
from std_msgs.msg import Float64

GPIO.setwarnings(False)

def get_distance():

    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    start = time.time()

    while GPIO.input(GPIO_ECHO)==0:
        start = time.time()

    while GPIO.input(GPIO_ECHO)==1:
	stop = time.time()

    elapsed = stop-start
    distance = (elapsed * 34300)/2

    return distance

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

while True:
	print(get_distance())
#	time.sleep(1)
