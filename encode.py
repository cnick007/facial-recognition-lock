from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Get the paths of the images to be encoded
imagePaths = sorted(list(paths.list_images('TestingData')))
faceEncodings = []
faceNames = []

for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    # Load the input image and convert it from BGR to RGB
    rgbImage = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)

    # Detect the faces and extract the encodings
    boxes = face_recognition.face_locations(rgbImage, model='hog')
    encodings = face_recognition.face_encodings(rgbImage, boxes)

    for encoding in encodings:
        faceEncodings.append(encoding)
        faceNames.append(name)

    # Only encode the first 10 faces for now...
    if i > 9:
        break

data = {"encodings": faceEncodings, "names": faceNames}

# Save encodings to a file for later use
if not os.path.exists("out"):
    os.mkdir("out")
f = open("out/face_enc", "wb")
f.write(pickle.dumps(data))
f.close()

print(data.get("names"))
