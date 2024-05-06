import cv2

# Function to detect accidents in video frames
def detect_accidents(video_path):
    cap = cv2.VideoCapture(video_path)  # Open video file or camera (0 for default camera)

    # Parameters for accident detection
    frame_count = 0
    prev_frame = None
    alert_threshold = 100 # Adjust this threshold based on your scene

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if video ends or error occurs

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        if prev_frame is None:
            prev_frame = blurred_frame
            continue

        # Compute absolute difference between current frame and previous frame
        frame_diff = cv2.absdiff(prev_frame, blurred_frame)
        diff_sum = frame_diff.sum()

        # Check if difference sum exceeds the threshold
        if diff_sum > alert_threshold:
            alert_and_capture(frame)  # Pass the current frame to alert function

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to display text alert and capture frame
def alert_and_capture(frame):
    print("ALERT: Accident Detected!")  # Display text alert

    # Save the frame where the accident is detected
    cv2.imwrite('accident_frame.jpg', frame)  # Corrected argument to frame, not Frame

# Example usage
if __name__ == "__main__":
    # Specify the video file path or use camera (0 for default camera)
    video_path = './dataset/video2.mp4'  # Replace with your video file path or use camera (0)

    # Perform accident detection
    detect_accidents(video_path)