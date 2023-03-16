from recognizer import Recognizer
import cv2
import os


def main():
    recognizer = Recognizer()

    if not os.path.exists("trainer/trainer.yml"):
        recognizer.train_new_face()

    print("Starting scan...")
    while True:
        cv2.imshow("Test", recognizer.recognize_faces(tolerance=25))

        # check if a face match was found
        if recognizer.face_match:
            print("Face Match!")

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


main()
