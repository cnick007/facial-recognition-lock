# facial-recognition-lock
## Installation
This was designed to work with a Raspberry Pi 4 and Pi Camera V2. The Pi Camera V2 may need to be setup in the Raspberry Pi config settings by enabling the Legacy Camera option.

Python3, Pip and venv are required, then run the following commands to get started:
```
git clone https://github.com/cnick007/facial-recognition-lock
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Once setup, run the main.py file. This will create a "KnownFaces" directory in which you can place any faces you would like to save and recognized. Face images should be placed in their own directories. The directories should be named the name of the person the face belongs to.

The hierarchy should look as follows (NOTE: image file names don't matter):
```
KnownFaces/Person1 Name/p1faceimage1.png
KnownFaces/Person1 Name/p1faceimage2.jpg

KnownFaces/Person2 Name/p2faceimage1.png
```

The more faces with the same name, the more accurate the recognition should be.

Once all desired faces are added to the "KnownFaces" directory, run the main.py file again. It should initialize and start scanning for known faces assuming the Pi Camera V2 is setup correctly. Once a known face is recognized, an out image will be placed in the "out" directory with the recognized face labeled.
