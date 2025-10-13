import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

# Ensure model and labels files exist
assert os.path.exists("keras_model.h5"), "Model file not found!"
assert os.path.exists("labels.txt"), "Labels file not found!"

# Initialize camera and modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

# Configuration parameters
offset = 20
imgSize = 300
counter = 0

# Updated labels with all the gestures
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
          "Thumbs up", "Thumbs down", "I love you", "Good luck", "Stop"]

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

                            # Ensure index is within bounds
                            if 0 <= index < len(labels):
                                predicted_label = labels[index]
                            else:
                                predicted_label = "Unknown"

                            print(f"Prediction: {prediction}, Index: {index}, Label: {predicted_label}")

                            # Calculate text width for proper rectangle sizing
                            text_size = cv2.getTextSize(predicted_label, cv2.FONT_HERSHEY_COMPLEX, 1.0, 2)[0]
                            rect_width = max(text_size[0] + 20, 90)

                            # Draw prediction rectangle and text
                            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                          (x - offset + rect_width, y - offset), (255, 0, 255), cv2.FILLED)
                            cv2.putText(imgOutput, predicted_label, (x - offset + 5, y - offset - 15),
                                        cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)

                            # Draw hand bounding box
                            cv2.rectangle(imgOutput, (x - offset, y - offset),
                                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

                            # Show cropped images
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
