# Hand Gesture Recognition

Lightweight hand-gesture recognition demo using OpenCV, cvzone's HandDetector and a Keras classification model.

---

## Overview

This repository contains a Python script that captures video from a webcam, detects a single hand using `cvzone.HandTrackingModule.HandDetector`, crops and resizes the detected hand region to a fixed `300x300` white canvas, and classifies the gesture with a Keras model (`keras_model.h5`) plus a `labels.txt` file. Predicted labels are shown on the video feed and the script displays the cropped and padded images in separate windows for debugging.

The script handles common runtime issues (camera open failure, empty crops, out-of-range label index, and prediction exceptions) and is suitable as a starting point for building real-time gesture-based controls or accessibility tools.

---

## Features

* Real-time hand detection (single hand) with `cvzone`.
* Robust cropping and aspect-ratio preserving resize into a `300x300` white canvas.
* Keras-based gesture classification via a `Classifier` wrapper.
* On-screen label drawing and crop preview windows.
* Basic error handling for camera, cropping, and prediction steps.

---

## Files

* `server.py` (or your main script): main application code (the provided script).
* `keras_model.h5`: trained Keras model (REQUIRED).
* `labels.txt`: newline-separated class labels that map indices to human-readable strings (REQUIRED).

> **Important:** The repository does **not** include the model or label file. Add your trained `keras_model.h5` and `labels.txt` to the repo root before running.

---

## Prerequisites

* Python 3.8+
* A webcam (script uses `cv2.VideoCapture(1)` — change the index to `0` if your webcam is at that device).

### Python packages

Install required packages (you can use `pip` or `venv`):

```bash
pip install opencv-python numpy pillow cvzone
```

**Notes:**

* `cvzone` depends on `mediapipe` internally; modern `cvzone` releases will pull it automatically. If you encounter issues, install `mediapipe` directly: `pip install mediapipe`.
* If your `Classifier` wrapper comes from `cvzone.ClassificationModule`, ensure you have the version of `cvzone` that provides it.

---

## Quick Start

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. Add your `keras_model.h5` and `labels.txt` to the project root.

3. (Optional) Update the webcam index in the script if necessary. The line to edit:

```python
cap = cv2.VideoCapture(1)
```

Change `1` to `0` or another index if your camera is at a different device.

4. Run the script:

```bash
python server.py
```

5. Use the camera window. Press `q` or `Esc` to quit.

---

## Configuration & Common Edits

* **`offset`**: number of pixels to pad the crop around the hand bounding box (default `20`).
* **`imgSize`**: canvas size used by the classifier (default `300`). If you trained your model with a different input size, update this accordingly.
* **`labels`**: the script contains an example `labels` list (A–J, thumbs up/down, etc.). Replace it with the contents of your `labels.txt` or load `labels.txt` at runtime to avoid mismatches.
* **Device index**: `cv2.VideoCapture(1)` — change to `0` on many laptops.

---

## Example `labels.txt` format

```
A
B
C
D
E
F
G
H
I
J
Thumbs up
Thumbs down
I love you
Good luck
Stop
```

Make sure the order of labels matches the class order used when training `keras_model.h5`.

---

## Troubleshooting

* **Camera won't open**: try different device indices (`0`, `1`, ...). Confirm another app can access the camera.
* **Model not found assertion**: ensure `keras_model.h5` is in the same folder as the script.
* **Label mismatch or index error**: verify `labels.txt` contains at least as many lines as the model outputs.
* **Black or blank `imgCrop`**: check bounding box coordinates and camera resolution; print `x,y,w,h` for debugging.
* **Slow performance**: reduce frame size, run at lower camera resolution, or run on a machine with GPU-enabled TensorFlow.

---

## Security & Privacy

This script captures video from your webcam. Be mindful of privacy: do not run on sensitive live feeds, and do not commit or share recordings containing personal data.

---

## Acknowledgements

* [cvzone](https://github.com/cvzone/cvzone) for the hand detection and utility classes.
* OpenCV and NumPy for image processing.

---

## Contact / Contribution

If you'd like improvements (e.g., automatic `labels.txt` loading, model training scripts, or UI enhancements), open an issue or submit a pull request.


