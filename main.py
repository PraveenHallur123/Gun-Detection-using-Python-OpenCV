import numpy as np
import cv2
import imutils
import datetime

# Load your trained cascade file
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Start webcam
camera = cv2.VideoCapture(0)

# Frame initialization
firstFrame = None
gun_exist = False

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect gun
    guns = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=20, minSize=(100, 100))

    if len(guns) > 0:
        gun_exist = True

    for (x, y, w, h) in guns:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Timestamp
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 0), 1)

    if gun_exist:
        print("🔫 Gun Detected!")
        cv2.imshow("Alert Frame", frame)
        cv2.waitKey(3000)  # Display alert for 3 seconds
        break
    else:
        cv2.imshow("Security Feed", frame)

    # Exit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
