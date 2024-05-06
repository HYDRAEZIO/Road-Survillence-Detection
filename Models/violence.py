import cv2
import numpy as np
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov5su.pt")

# Define the violence classes
violence_classes = ['gun', 'knife', 'baseball bat', 'fight']

# Open the default camera
video_path='./dataset/video6.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform object detection on the frame
    results = model.predict(frame)

    # Iterate through the detected objects
    for result in results:
        for box in result.boxes:
            # Get the class ID and confidence score
            class_id = box.cls[0].item()
            confidence = box.conf[0].item()

            # Check if the object is a violence-related object
            if class_id in violence_classes and confidence > 0.5:
                # Draw a bounding box around the object
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Display an alert message
                cv2.putText(frame, "VIOLENCE DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()