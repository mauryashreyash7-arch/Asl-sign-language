import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

# Optional: reference to your screenshot (local)
# screenshot_url = "sandbox:/mnt/data/7a1b9233-ddbb-4ba4-9bbb-5f75552f8649.png"

# Ensure model and labels files exist
assert os.path.exists("keras_model.h5"), "Model file not found!"
assert os.path.exists("labels.txt"), "Labels file not found!"

# Initialize camera and modules
cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

# Configuration parameters
offset = 20
imgSize = 300
counter = 0

# Updated labels (0–14)
labels = [
    "A",            # 0
    "B",            # 1
    "C",            # 2
    "D",            # 3
    "E",            # 4
    "F",            # 5
    "G",            # 6
    "H",            # 7
    "I",            # 8
    "J",            # 9
    "Thumbs up",    # 10
    "Thumbs down",  # 11
    "I love you",   # 12
    "Good luck",    # 13
    "Stop"          # 14
]

# Nice accent color (teal) for badges and boxes
ACCENT_COLOR = (0, 153, 204)  # BGR (blue, green, red)
TEXT_COLOR = (255, 255, 255)  # white

# Check if camera is opened
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

try:
    while True:
        success, img = cap.read()

        # Check if frame was read successfully
        if not success:
            print("Error: Failed to read from camera")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Create bounds checking to prevent index errors
            img_height, img_width = img.shape[:2]

            # Calculate crop boundaries with safety checks
            y_start = max(0, y - offset)
            y_end = min(img_height, y + h + offset)
            x_start = max(0, x - offset)
            x_end = min(img_width, x + w + offset)

            # Check if crop area is valid
            if y_end > y_start and x_end > x_start:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y_start:y_end, x_start:x_end]

                # Check if crop is not empty
                if imgCrop.size > 0:
                    imgCropShape = imgCrop.shape

                    # Prevent division by zero
                    if w > 0 and h > 0:
                        aspectRatio = h / w
                        try:
                            if aspectRatio > 1:
                                k = imgSize / h
                                wCal = math.ceil(k * w)
                                if wCal > 0:
                                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                                    wGap = math.ceil((imgSize - wCal) / 2)
                                    imgWhite[:, wGap:wCal + wGap] = imgResize
                            else:
                                k = imgSize / w
                                hCal = math.ceil(k * h)
                                if hCal > 0:
                                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                                    hGap = math.ceil((imgSize - hCal) / 2)
                                    imgWhite[hGap:hCal + hGap, :] = imgResize

                            # Get prediction with error handling
                            prediction, index = classifier.getPrediction(imgWhite, draw=False)

                            # Attempt to extract a confidence score (robust to common output shapes)
                            confidence = None
                            try:
                                # case: prediction is list of probs
                                if isinstance(prediction, (list, tuple, np.ndarray)):
                                    # convert numpy arrays to list
                                    if isinstance(prediction, np.ndarray):
                                        arr = prediction.tolist()
                                    else:
                                        arr = list(prediction)
                                    # if numeric list -> take max as confidence
                                    if arr and all(isinstance(x, (int, float, np.floating, np.integer)) for x in arr):
                                        confidence = float(max(arr))
                                    # if prediction like ['label', 0.9]
                                    elif len(arr) >= 2 and isinstance(arr[1], (int, float)):
                                        confidence = float(arr[1])
                                elif isinstance(prediction, dict) and "confidence" in prediction:
                                    confidence = float(prediction["confidence"])
                            except Exception:
                                confidence = None

                            # Ensure index is within bounds
                            if 0 <= index < len(labels):
                                predicted_label = labels[index]
                            else:
                                predicted_label = "Unknown"

                            # Format confidence text
                            conf_text = ""
                            if confidence is not None:
                                # Clamp and convert to percentage
                                try:
                                    c = float(confidence)
                                    if c <= 1.5:  # likely already 0-1
                                        pct = max(0.0, min(100.0, c * 100.0))
                                    else:
                                        # already a large number? scale defensively
                                        pct = max(0.0, min(100.0, c))
                                    conf_text = f"{pct:.0f}%"
                                except Exception:
                                    conf_text = ""

                            # Compose the display text
                            if conf_text:
                                display_text = f"{predicted_label}  {conf_text}"
                            else:
                                display_text = f"{predicted_label}"

                            print(f"Prediction: {prediction}, Index: {index}, Label: {predicted_label}, Confidence: {conf_text}")

                            # Calculate text width for proper rectangle sizing
                            font = cv2.FONT_HERSHEY_COMPLEX
                            font_scale = 0.9
                            thickness = 2
                            (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                            rect_padding_x = 14
                            rect_padding_y = 10
                            rect_width = text_w + rect_padding_x * 2
                            rect_height = text_h + rect_padding_y * 2

                            # Position the rectangle above the hand (or clamp to top)
                            rect_x1 = max(0, x - offset)
                            rect_y1 = max(0, y - offset - rect_height - 10)
                            rect_x2 = min(img_width, rect_x1 + rect_width)
                            rect_y2 = rect_y1 + rect_height

                            # Draw filled rectangle (accent color) and put text
                            cv2.rectangle(imgOutput, (rect_x1, rect_y1), (rect_x2, rect_y2), ACCENT_COLOR, cv2.FILLED)
                            text_x = rect_x1 + rect_padding_x
                            text_y = rect_y1 + rect_padding_y + text_h
                            cv2.putText(imgOutput, display_text, (text_x, text_y), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)

                            # Draw hand bounding box using the same accent color (thicker for nicer look)
                            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), ACCENT_COLOR, 3)

                            # Optionally, draw a smaller confidence bar below the rectangle (visual cue)
                            if confidence is not None:
                                # small horizontal bar showing confidence percentage
                                try:
                                    # compute pct 0-1
                                    c = float(confidence)
                                    pct = (c if c <= 1.5 else c/100.0)
                                    pct = max(0.0, min(1.0, pct))
                                    bar_w = int((rect_x2 - rect_x1) * pct)
                                    bar_h = 6
                                    bar_x1 = rect_x1
                                    bar_y1 = rect_y2 + 6
                                    bar_x2 = rect_x1 + bar_w
                                    bar_y2 = bar_y1 + bar_h
                                    # background bar
                                    cv2.rectangle(imgOutput, (rect_x1, bar_y1), (rect_x1 + (rect_x2 - rect_x1), bar_y2), (220, 220, 220), cv2.FILLED)
                                    # foreground bar (darker accent)
                                    cv2.rectangle(imgOutput, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 120, 170), cv2.FILLED)
                                except Exception:
                                    pass

                            # Show cropped images (optional; helpful for debugging)
                            cv2.imshow("ImageCrop", imgCrop)
                            cv2.imshow("ImageWhite", imgWhite)

                        except Exception as e:
                            print(f"Error in prediction: {e}")
                            continue

        # Display main output
        cv2.imshow("Hand Gesture Recognition", imgOutput)

        # Break loop on 'q' key press or window close
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or Escape key
            break

        # Check if window was closed
        if cv2.getWindowProperty("Hand Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended successfully")
