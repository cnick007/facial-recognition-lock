import cv2
import numpy as np
import os


class Recognizer:
    def __init__(self) -> None:
        print("Initializing recognizer...")

        self.face_match = False
        # Load the classifier
        self.cv2_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Init the cv2 recognizer
        self.cv2_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Init Camera
        self.cv2_capture = cv2.VideoCapture(0)
        if not self.cv2_capture:
            print("Failed to initialize camera!")
            exit(1)
        self.cv2_capture.set(3, 640)  # set video width
        self.cv2_capture.set(4, 480)  # set video height

        if not os.path.exists("trainer"):
            os.mkdir("trainer")

        if not os.path.exists("trainer/trainer.yml"):
            # we need to ensure that there are known faces
            print("trainer.yml not found.")
            self.train_new_face()

        self.cv2_recognizer.read("trainer/trainer.yml")

    def __del__(self):
        print("Deinitializing recognizer...")
        # Release the capture
        self.cv2_capture.release()

    def recognize_faces(self, tolerance=25):
        """Updates the current frame and scans for known faces using a tolerance.

        Args:
            tolerance: the tolerance that is compared with the confidence of the face detected
            frame: the frame to scan for faces

        Returns:
            a modified frame outlining the scanned face with a confidence value
        """

        ret, frame = self.cv2_capture.read()
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.cv2_face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        self.face_match = False
        if len(faces) == 0:
            return frame

        # Loop through detected faces
        for x, y, w, h in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Predict the label for the detected face
            label, confidence = self.cv2_recognizer.predict(gray[y : y + h, x : x + w])

            # If the confidence is high enough, display the name
            if confidence < tolerance:
                self.face_match = True
            else:
                self.face_match = False
            cv2.putText(
                frame,
                str(round(confidence, 2)),
                (x + 150, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        return frame

    def train_new_face(self):
        """Trains a new face to be saved to trainer/trainer.yml. Call this once and it will scan until 30 face samples are collected."""
        print("Starting training of recognizer...")
        # Initialize lists for face samples and labels
        samples = []
        labels = []

        # Loop until we have enough face samples
        num_samples = 0
        while num_samples < 30:
            # Capture frame-by-frame
            ret, frame = self.cv2_capture.read()

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.cv2_face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            # Loop through detected faces
            for x, y, w, h in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add face sample and label to lists
                samples.append(gray[y : y + h, x : x + w])
                labels.append(
                    1
                )  # Change this to the label you want to use for this person

                # Increment the number of samples we've collected
                num_samples += 1

            # Display the resulting frame
            cv2.imshow("Training", frame)

            # Wait for 100ms or until 'q' is pressed to exit
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        # Train the face recognition model
        self.cv2_recognizer.train(samples, np.array(labels))

        # Save the trained model to a file
        self.cv2_recognizer.write("trainer/trainer.yml")
        cv2.destroyAllWindows()
