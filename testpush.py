from gpiozero import Servo
from time import sleep
import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

while True:
    if GPIO.input(25) == GPIO.HIGH:
        print("Button at GPIO 25 was pushed!")
        sleep(1)
    if GPIO.input(27) == GPIO.HIGH:
        print("Button at GPIO 27 was pushed!")
        sleep(1)
