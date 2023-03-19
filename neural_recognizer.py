import cv2
import os
import face_recognition


class Recognizer:
    def __init__(self):
        print("Initializing recognizer...")

        self.face_match = False
        self.process_frame = True
        self.cv2_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Init Camera
        self.cv2_capture = self.init_camera()

        if not os.path.exists("known_faces"):
            os.mkdir("known_faces")
            self.train_new_face()

        # Load the known faces
        self.known_face_encodings = self.load_known_faces("known_faces")

    def __del__(self):
        print("Deinitializing recognizer...")
        # Release the capture
        self.cv2_capture.release()

    @staticmethod
    def init_camera():
        # Init Camera
        cv2_capture = cv2.VideoCapture(0)
        if not cv2_capture.isOpened():
            print("Failed to initialize camera!")
            exit(1)
        cv2_capture.set(3, 640)  # set video width
        cv2_capture.set(4, 480)  # set video height
        return cv2_capture

    @staticmethod
    def load_known_faces(folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

        known_images = []
        for filename in os.listdir(folder):
            img = face_recognition.load_image_file(os.path.join(folder, filename))
            if img is not None:
                known_images.append(img)

        known_face_encodings = []
        for image in known_images:
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) > 0:
                known_face_encodings.extend(face_encoding)
        return known_face_encodings

    def recognize_faces(self):
        self.face_match = False
        ret, frame = self.cv2_capture.read()

        if not self.process_frame:
            self.process_frame = not self.process_frame
            return frame

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cv2_face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )
        if len(faces) == 0:
            return frame

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        unknown_face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations
        )

        if len(face_locations) == 0:
            return frame

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, unknown_face_encodings
        ):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, 0.3
            )

            text = "Unknown"
            if True in matches:
                self.face_match = True
                text = "Known"

            cv2.putText(
                frame,
                text,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return frame

    def train_new_face(self):
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
                samples.append(frame[y : y + h, x : x + w])
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
        for i, img in enumerate(samples):
            cv2.imwrite("known_faces/img-{0}.png".format(i), img)

        cv2.destroyAllWindows()
