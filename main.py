from neural_recognizer import Recognizer
import cv2
from time import time
from time import sleep
import RPi.GPIO as GPIO
import os

delay = 10
locked = True


def unlock():
    global locked
    locked = False
    GPIO.output(18, 1)
    print("Unlock door!")
    return time()


def lock():
    global locked
    locked = True
    print("Lock door!")
    GPIO.output(18, 0)


def main():
    unlock_time = 0

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(25, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(18, GPIO.OUT)
    GPIO.output(18, 0)

    faces_folder = "known_faces"
    tolerance = 0.4
    recognizer = Recognizer(faces_folder, tolerance)
    if not os.path.exists(faces_folder):
        os.mkdir(faces_folder)
        recognizer.train_new_face()

    recognizer.load_known_faces()

    print("Starting scan...")
    lock()
    while True:
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        cv2.imshow("Face Recognition", recognizer.recognize_faces())

        if GPIO.input(25) == GPIO.HIGH:
            print("Adding new face!")
            recognizer.train_new_face()
            recognizer.load_known_faces()
            sleep(1)

        if time() > (unlock_time + delay):
            if not locked:
                lock()
            # check if a face match was found
            if recognizer.face_match:
                unlock_time = unlock()
                print("Face Match!")

            if GPIO.input(27) == GPIO.HIGH:
                print("Emergency unlock!")
                unlock_time = unlock()
                sleep(1)


main()
