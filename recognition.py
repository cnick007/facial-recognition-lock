import face_recognition
import pickle
import cv2
import os

# Load the cascade classifier
faceCascade = cv2.CascadeClassifier(
    os.getcwd() + '/Classifiers/frontal_haarcascade_classifier.xml')
# Load the known faces
knownFaces = pickle.loads(open('out/face_enc', "rb").read())
# Load the image with the face we want to detect
inputImage = cv2.imread("img.jpg")
rgbImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Detect the faces within the image
faces = faceCascade.detectMultiScale(grayImage,
                                     scaleFactor=1.1,
                                     minNeighbors=9,
                                     minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

# Get the encodings of the faces in the input image
encodings = face_recognition.face_encodings(rgbImage)
names = []

for encoding in encodings:
    # Compare input image encodings with the known face encodings
    matches = face_recognition.compare_faces(knownFaces["encodings"],
                                             encoding)

    name = "Unknown"
    # Get the indexes of each face match
    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    count = {}

    for i in matchedIdxs:
        # Check the names at respective indexes we stored in matchedIdxs
        name = knownFaces["names"][i]
        # Increase count for the name we got
        count[name] = count.get(name, 0) + 1
        name = max(count, key=count.get)
    # Update the list of names
    names.append(name)

# Loop over the recognized faces
for ((x, y, w, h), name) in zip(faces, names):
    # Rescale the face coordinates
    # Draw the predicted face name on the image
    cv2.rectangle(inputImage, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(inputImage, name, (x, y - 4), cv2.FONT_ITALIC,
                0.75, (0, 0, 255), 2)
    if not os.path.exists("out"):
        os.mkdir("out")
    cv2.imwrite("out/out_img.jpg", inputImage)
    cv2.waitKey(0)
