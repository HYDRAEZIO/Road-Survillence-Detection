import cv2
import time

# Function to calculate speed
def calculate_speed(start_time, end_time, distance_in_meters):
    time_diff = end_time - start_time
    if time_diff != 0:
        speed = distance_in_meters / time_diff  # Speed = Distance / Time
        return speed
    else:
        return 0  # Return 0 if time difference is 0 (to avoid division by zero)

# Function to raise alert if speed crosses threshold
def raise_alert(speed_threshold, current_speed):
    if current_speed > speed_threshold:
        print(f"ALERT: Speed Limit Exceeded! Current Speed: {current_speed} fps")

# Main function for speed detection
def car_speed_detection(video_path):
    cap = cv2.VideoCapture(video_path)  # Open video file or camera (0 for default camera)

    # Parameters for speed calculation and alert
    frame_count = 0
    start_time = time.time()  # Initialize start_time
    distance_covered_in_pixels = 0  # Adjust this based on your camera setup
    speed_threshold = 60  # Speed threshold for alert in km/h

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if video ends or error occurs

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred_frame, 50, 150)

        # Detect edges and calculate distance covered
        frame_diff = cv2.absdiff(start_time, edges)
        contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Adjust this threshold based on your scene
                distance_covered_in_pixels += cv2.arcLength(contour, True)

        # Calculate speed every few frames
        frame_count += 1
        if frame_count == 10:  # Calculate speed every 10 frames (adjust as needed)
            end_time = time.time()
            current_speed = calculate_speed(start_time, end_time, distance_covered_in_pixels)
            print(f"Current Speed: {current_speed} fps")  # Display speed
            raise_alert(speed_threshold, current_speed)  # Check and raise alert
            frame_count = 0
            start_time = time.time()

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = './dataset/video1.avi'  # Replace with your video file path or use camera (0)
car_speed_detection(video_path)