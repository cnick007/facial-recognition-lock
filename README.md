# facial-recognition-lock
## Installation
This was designed to work with a Raspberry Pi 4 and Pi Camera V2. The Pi Camera V2 may need to be setup in the Raspberry Pi config settings by enabling the Legacy Camera option.

Python3, Pip and venv are required, then run the following commands to get started:
```
git clone https://github.com/cnick007/facial-recognition-lock
cd facial-recognition-lock
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
### neural_recognizer
Once setup, run the main.py file. This will create a "known_faces" directory, launch a window showing the camera, and start scanning for a face to add to the recognizer. After it has collected 10 samples, the faces will be saved to "known_faces/img-#.png". It will then launch another window that scans for known faces. If a match is found, it will print "Face Match!" in the console.

The system utilizes GPIO 25 and 27 to handle adding new faces and emergency unlock respectively. To utilize this functionality, the system will need to be have some form of input (push buttons preferably) hooked up to these GPIO pins on the Pi 4. The system also controls a locking mechanism that will need to be attached to GPIO 18 in the form of a solenoid lock.

Once the hardware is connected, the system should now be a fully functional facial recognition locking system. To add new faces, press the button connected to GPIO 25. To open the lock in an emergency, press the secret emergency unlock button at GPIO 27. Whenever a known face is detected in the camera frame, the door will unlock for 10s giving the user enough time to open the door before locking again. 
