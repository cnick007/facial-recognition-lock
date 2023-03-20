# Import all neccessary features to code.
import RPi.GPIO as GPIO
from time import sleep

# If code is stopped while the solenoid is active it stays active
# This may produce a warning if the code is restarted and it finds the GPIO Pin, which it defines as non-active in next line, is still active
# from previous time the code was run. This line prevents that warning syntax popping up which if it did would stop the code running.
GPIO.setwarnings(False)
# This means we will refer to the GPIO pins
# by the number directly after the word GPIO. A good Pin Out Resource can be found here https://pinout.xyz/
GPIO.setmode(GPIO.BCM)
# This sets up the GPIO 18 pin as an output pin
GPIO.setup(18, GPIO.OUT)

while True:
    # This Turns Relay Off. Brings Voltage to Max GPIO can output ~3.3V
    GPIO.output(18, 1)
    # Wait 1 Seconds
    sleep(1)
    # Turns Relay On. Brings Voltage to Min GPIO can output ~0V.
    GPIO.output(18, 0)
    # Wait 1 Seconds
    sleep(1)
