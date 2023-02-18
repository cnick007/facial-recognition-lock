import cv2
import os
import face_recognition
import pickle
from imutils import paths


def GetDetectedFaces(image):
    """Detects and finds the positions of all faces in the image

    Parameters:
    image: the image to detect faces in

    Returns:
    an array of (x, y, width, height) outlining the faces
    """
    # Load the cascade classifier
    faceCascade = cv2.CascadeClassifier(
        os.getcwd() + "/Classifiers/frontal_haarcascade_classifier.xml"
    )
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the faces within the image
    return faceCascade.detectMultiScale(
        grayImage,
        scaleFactor=1.1,
        minNeighbors=9,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


def GetFaceEncodings(image):
    """Encodes and returns all faces in an image

    Parameters:
        image: the image to get the encodings of

    Returns:
        list: encodings of all faces in the image
    """
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return face_recognition.face_encodings(rgbImage)


def GetKnownFaceEncodings(encPath):
    """Retrieves the saved face encodings

    Args:
        encPath: the path to the face encodings file

    Returns:
        list: a list of saved face encodings
    """
    return pickle.loads(open(encPath, "rb").read())


def ContainsKnownFace(image, encPath):
    """Checks if an image contains a known face

    Args:
        image: the image to check
        encPath: the path to the face encodings file

    Returns:
        bool: true if it contains a known face, false otherwise
    """
    knownFaces = GetKnownFaceEncodings(encPath)
    encodings = GetFaceEncodings(image)
    matchFound = False

    for encoding in encodings:
        matches = face_recognition.compare_faces(knownFaces["encodings"], encoding)
        name = "Unknown"
        match = False
        for i, match in enumerate(matches):
            if match:
                matchFound = True
                name = knownFaces["names"][i]
        if match:
            print("Known Face Found: ", name)

    return matchFound


def IdentifyAndLabelFace(image, encPath):
    """Identifies a face in the image and labels it with its accociated name (debugging)

    Args:
        image: the image to label

    Returns:
        image: a labled version of the input image
        encPath: path to the known faces encodings
    """
    knownFaces = GetKnownFaceEncodings(encPath)
    encodings = GetFaceEncodings(image)
    faces = GetDetectedFaces(image)
    names = []

    for encoding in encodings:
        # Compare input image encodings with the known face encodings
        matches = face_recognition.compare_faces(knownFaces["encodings"], encoding)

        name = "Unknown"
        # Get the indexes of each face match
        matchedIdxs = [i for (i, match) in enumerate(matches) if match]
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
    for (x, y, w, h), name in zip(faces, names):
        # Rescale the face coordinates
        # Draw the predicted face name on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, name, (x, y - 4), cv2.FONT_ITALIC, 0.75, (0, 0, 255), 2)
    return image


def EncodeFaces(facesDir, encPath):
    """Encodes the faces in the facesDir to encPath

    Args:
        encPath: path to save the encodings to
        facesDir: directory conataining the faces to encode
    """
    # Get the paths of the images to be encoded
    imagePaths = sorted(list(paths.list_images(facesDir)))
    faceEncodings = []
    faceNames = []

    for i, imagePath in enumerate(imagePaths):
        # Extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]
        # Load the input image and convert it from BGR to RGB
        rgbImage = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)

        # Detect the faces and extract the encodings
        boxes = face_recognition.face_locations(rgbImage, model="hog")
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
    f = open(encPath, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("Encoding faces in: " + facesDir + " to: " + encPath)
    print(set(data.get("names")))


def main():
    print("Initializing...")
    knownFacesDir = "KnownFaces"
    encPath = "out/face_enc"
    capture = None

    if not os.path.exists("out"):
        os.mkdir("out")
    if not os.path.exists("KnownFaces"):
        os.mkdir("KnownFaces")
    print(
        "Place any faces you want saved into the KnownFaces directory organized in folders by name."
    )

    if not os.path.exists(encPath):
        EncodeFaces(knownFacesDir, encPath)

    try:
        capture = cv2.VideoCapture(0)
    except:
        print("Failed to open camera")
        exit()

    capture.set(3, 640)
    capture.set(4, 480)

    print("Started capturing")
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error reading video")
            break

        if ContainsKnownFace(frame, encPath):
            cv2.imwrite("out/out_img.jpg", IdentifyAndLabelFace(frame, encPath))

    capture.release()


if __name__ == "__main__":
    main()
