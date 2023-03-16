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
Once setup, run the main.py file. This will create a "trainer" directory, launch a window showing the camera, and start scanning for a face to add to the recognizer. After it has collected 30 samples, the trained face will be saved to "trainer/trainer.yml". It will then launch another window that scans for known faces and will outline all faces in the frame and give them a confidence rating (lower = better). If the confidence is below a tolerance of 25, the face is considered known and "Face Match!" is printed to the console.
