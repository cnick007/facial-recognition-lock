import cv2
import os
import face_recognition


class Recognizer:
    def __init__(self, faces_folder, tolerance):
        print("Initializing recognizer...")

        self.tolerance = tolerance
        self.known_face_encodings = []
        self.faces_folder = faces_folder
        self.face_match = False
        self.process_frame = True
        self.cv2_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Init Camera
        self.cv2_capture = self.init_camera()

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

    def load_known_faces(self):
        if not os.path.exists(self.faces_folder):
            os.mkdir(self.faces_folder)

        known_images = []
        for filename in os.listdir(self.faces_folder):
            img = face_recognition.load_image_file(
                os.path.join(self.faces_folder, filename)
            )
            if img is not None:
                known_images.append(img)

        for image in known_images:
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) > 0:
                self.known_face_encodings.extend(face_encoding)

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
        face_locations = face_recognition.face_locations(rgb_frame)
        unknown_face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations
        )

        if len(face_locations) == 0:
            return frame

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, unknown_face_encodings
        ):
            # rescale up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, self.tolerance
            )

            text = "Unknown"
            color = (255, 0, 0)
            if True in matches:
                self.face_match = True
                text = "Known"
                color = (0, 255, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 5)
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
        while num_samples < 10:
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

                # Add face sample and label to lists
                samples.append(frame[y : y + h, x : x + w])
                labels.append(
                    1
                )  # Change this to the label you want to use for this person

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Increment the number of samples we've collected
                num_samples += 1

            # Display the resulting frame
            cv2.imshow("Training", frame)

            # Wait for 100ms or until 'q' is pressed to exit
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
        count = len(
            [
                entry
                for entry in os.listdir(self.faces_folder)
                if os.path.isfile(os.path.join(self.faces_folder, entry))
            ]
        )
        # Train the face recognition model
        for i, img in enumerate(samples):
            cv2.imwrite(self.faces_folder + "/img-{0}.png".format(i + count), img)

        cv2.destroyAllWindows()
