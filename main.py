from neural_recognizer import Recognizer
import cv2
from time import sleep
import RPi.GPIO as GPIO


def main():
    recognizer = Recognizer()
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)
    GPIO.output(18, 1)

    print("Starting scan...")
    while True:
        cv2.imshow("Test", recognizer.recognize_faces())

        # check if a face match was found
        if recognizer.face_match:
            GPIO.output(18, 0)
            print("Face Match!")
            sleep(10)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


main()
