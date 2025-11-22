# server.py
from flask import Flask, Response, send_from_directory, jsonify, request
import cv2
import numpy as np
import os
import math
import time
import threading

# import your cvzone modules
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# --- config ---
MODEL_PATH = 'keras_model.h5'
LABELS_PATH = 'labels.txt'
OFFSET = 20
IMG_SIZE = 300
CAM_INDEX = 0   # change to 1 if your camera is at index 1

# check files
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Put keras_model.h5 and labels.txt next to server.py")

with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f if line.strip()]

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Camera and model initialization
cap = cv2.VideoCapture(CAM_INDEX)
detector = HandDetector(maxHands=1)
classifier = Classifier(MODEL_PATH, LABELS_PATH)

# Shared state
last_prediction = {'prediction': 'No hand detected'}
last_prediction_lock = threading.Lock()
recognition_active = True
recognition_flag_lock = threading.Lock()

def set_last_prediction(text):
    with last_prediction_lock:
        last_prediction['prediction'] = text

def get_last_prediction():
    with last_prediction_lock:
        return last_prediction['prediction']

def set_recognition_active(val: bool):
    global recognition_active
    with recognition_flag_lock:
        recognition_active = bool(val)

def get_recognition_active():
    with recognition_flag_lock:
        return recognition_active

def process_frame_and_predict(frame):
    """
    Detect hand, crop, resize, predict and return (frame_with_drawings, prediction_text or None)
    If recognition is not active, prediction will be None.
    """
    imgOutput = frame.copy()
    hands, _ = detector.findHands(frame, draw=False)
    predicted_label = None

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        img_h, img_w = frame.shape[:2]

        y_start = max(0, y - OFFSET)
        y_end = min(img_h, y + h + OFFSET)
        x_start = max(0, x - OFFSET)
        x_end = min(img_w, x + w + OFFSET)

        if y_end > y_start and x_end > x_start:
            imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
            imgCrop = frame[y_start:y_end, x_start:x_end]

            if imgCrop.size > 0 and w > 0 and h > 0:
                aspectRatio = h / w
                try:
                    if aspectRatio > 1:
                        k = IMG_SIZE / h
                        wCal = math.ceil(k * w)
                        if wCal > 0:
                            imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                            wGap = math.ceil((IMG_SIZE - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = IMG_SIZE / w
                        hCal = math.ceil(k * h)
                        if hCal > 0:
                            imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                            hGap = math.ceil((IMG_SIZE - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize

                    if get_recognition_active():
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        if 0 <= index < len(labels):
                            predicted_label = labels[index]
                        else:
                            predicted_label = 'Unknown'
                        set_last_prediction(predicted_label)

                    # draw UI on frame
                    label_to_draw = get_last_prediction()
                    text_size = cv2.getTextSize(label_to_draw, cv2.FONT_HERSHEY_COMPLEX, 1.0, 2)[0]
                    rect_width = max(text_size[0] + 20, 90)
                    cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET - 50), (x - OFFSET + rect_width, y - OFFSET), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, label_to_draw, (x - OFFSET + 5, y - OFFSET - 15), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255), 2)
                    cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), (255,0,255), 4)

                except Exception as e:
                    print("Prediction error:", e)

    return imgOutput, predicted_label

def gen_frames():
    """Yield MJPEG frames to client"""
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame, _ = process_frame_and_predict(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # multipart/x-mixed-replace stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify({'prediction': get_last_prediction()})

@app.route('/start', methods=['POST'])
def start_recognition():
    set_recognition_active(True)
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop_recognition():
    set_recognition_active(False)
    return jsonify({'status': 'stopped'})

# graceful shutdown: release camera when exiting
def cleanup():
    try:
        cap.release()
    except:
        pass

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        cleanup()

