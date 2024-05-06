import cv2
import numpy as np
import imutils
import pytesseract
import pandas as pd
import time

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model and class labels for violence and weapon detection
net_violence = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net_weapon = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes_violence = [line.strip() for line in open("coco.names").readlines()]
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classes_weapon = ["weapon"]  # Adjust class as needed for weapon detection


# Function to detect number plate in an image
def detect_number_plate(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 170, 200)

    # Check OpenCV version to handle findContours return values
    cv_version = cv2.__version__.split('.')[0]
    if cv_version == '3':
        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    elif cv_version == '4':
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    # Configuration for tesseract
    config = ('-l eng --oem 1 --psm 3')

    # Run tesseract OCR on image
    text = pytesseract.image_to_string(new_image, config=config)

    # Data is stored in CSV file
    raw_data = {'date': [time.asctime(time.localtime(time.time()))],
                'v_number': [text]}
    df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
    df.to_csv('data.csv')

    return text

# Function to detect accidents in video frames
def detect_accidents(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    prev_frame = None
    alert_threshold = 100

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        if prev_frame is None:
            prev_frame = blurred_frame
            continue

        frame_diff = cv2.absdiff(prev_frame, blurred_frame)
        diff_sum = frame_diff.sum()

        if diff_sum > alert_threshold:
            alert_and_capture(frame)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def alert_and_capture(frame):
    print("ALERT: Accident Detected!")
    cv2.imwrite('accident_frame.jpg', frame)

# Function to calculate speed
def calculate_speed(start_time, end_time, distance_in_meters):
    time_diff = end_time - start_time
    if time_diff != 0:
        speed = distance_in_meters / time_diff  # Speed = Distance / Time
        return speed
    else:
        return 0

# Function to raise alert if speed crosses threshold
def raise_alert(speed_threshold, current_speed):
    if current_speed > speed_threshold:
        print(f"ALERT: Speed Limit Exceeded! Current Speed: {current_speed} km/h")

# Function to perform violence detection
def detect_violence(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_violence.setInput(blob)
    outs = net_violence.forward(net_violence.getUnconnectedOutLayersNames())

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes_violence[class_id] == "person":
                return True  # Violence detected (person detected)

    return False  # No violence detected

# Function to perform weapon detection
def detect_weapon(frame):
    height, width, _ = frame.shape
    target_width = 416
    target_height = int(height * (target_width / width))
    resized_frame = cv2.resize(frame, (target_width, target_height))
    top_pad = (416 - target_height) // 2
    bottom_pad = 416 - target_height - top_pad
    padded_frame = cv2.copyMakeBorder(resized_frame, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    blob = cv2.dnn.blobFromImage(padded_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_weapon.setInput(blob)
    outs = net_weapon.forward(net_weapon.getUnconnectedOutLayersNames())

    weapon_detected = False  # Initialize flag for weapon detection

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes_weapon and class_id < len(classes_weapon) and confidence > 0.5:  # Check confidence threshold
                if classes_weapon[class_id] == "weapon":
                    weapon_detected = True  # Set flag if weapon is detected

    return weapon_detected  # Return the weapon detection flag

# Main function to process video and perform detections
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()
    distance_covered_in_pixels = 0
    speed_threshold = 60

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if detect_violence(frame):
            print("ALERT: Violence Detected!")

        if detect_weapon(frame):
            print("ALERT: Weapon Detected!")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred_frame, 50, 150)

        frame_diff = cv2.absdiff(start_time, edges)
        contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                distance_covered_in_pixels += cv2.arcLength(contour, True)

        frame_count += 1
        if frame_count == 10:
            end_time = time.time()
            current_speed = calculate_speed(start_time, end_time, distance_covered_in_pixels)
            print(f"Current Speed: {current_speed} km/h")
            raise_alert(speed_threshold, current_speed)
            frame_count = 0
            start_time = time.time()

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = "./dataset/video1.avi"  # Replace with your video file path or use camera (0)
    main(0)