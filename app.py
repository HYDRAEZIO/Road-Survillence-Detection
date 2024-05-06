import time
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from ultralytics import YOLO
import PIL

class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Survillence")

        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        self.open_button = tk.Button(root, text="Open Video", command=self.open_video)
        self.open_button.pack(pady=5)

        self.play_button = tk.Button(root, text="Play", command=self.play_video)
        self.play_button.pack(pady=5)

        self.pause_button = tk.Button(root, text="Pause", command=self.pause_video, state=tk.DISABLED)
        self.pause_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_video, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.violence_button = tk.Button(root, text="Violence", command=self.detect_violence)
        self.violence_button.pack(pady=5)

        self.car_speed_button = tk.Button(root, text="Car Speed", command=self.detect_car_speed)
        self.car_speed_button.pack(pady=5)

        self.car_accident_button = tk.Button(root, text="Car Accident", command=self.detect_accidents)
        self.car_accident_button.pack(pady=5)

        self.weapon_button = tk.Button(root, text="Weapon Detection", command=self.detect_weapons)
        self.weapon_button.pack(pady=5)

        self.video_capture = None
        self.is_playing = False
        self.model = YOLO("yolov5su.pt")
        self.violence_classes = ['gun', 'knife', 'baseball bat', 'fight']

    def open_video(self):
        video_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if video_file:
            self.video_capture = cv2.VideoCapture(video_file)

    def play_video(self):
        if self.video_capture:
            self.is_playing = True
            self.play_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.play_video_frame()

    def play_video_frame(self):
        if self.is_playing and self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))  # Resize for display if needed
                frame = self.detect_violence_in_frame(frame)  # Call violence detection
                frame = PIL.Image.fromarray(frame)
                frame = PIL.Image.PhotoImage(frame)
                self.video_label.config(image=frame)
                self.video_label.image = frame
                self.root.after(30, self.play_video_frame)
            else:
                self.stop_video()

    def pause_video(self):
        self.is_playing = False
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)

    def stop_video(self):
        self.is_playing = False
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.video_capture.release()
        self.video_label.config(image="")

    def detect_violence(self):
        if self.video_capture:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                frame = self.detect_violence_in_frame(frame)
                
                cv2.imshow("Violence Detection", frame)
                print('violence detected')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def detect_violence_in_frame(self, frame):
        results = self.model.predict(frame)
        for result in results:
            for box in result.boxes:
                class_id = box.cls[0].item()
                confidence = box.conf[0].item()
                if class_id in self.violence_classes and confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "VIOLENCE DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    def detect_car_speed(self):
        if self.video_capture:
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
                    print(f"ALERT: Speed Limit Exceeded! Current Speed: {current_speed} km/h")

            # Main function for speed detection
            start_time = time.time()  # Initialize start_time
            distance_covered_in_pixels = 0  # Adjust this based on your camera setup
            speed_threshold = 60  # Speed threshold for alert in km/h

            while True:
                ret, frame = self.video_capture.read()
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
                end_time = time.time()
                current_speed = calculate_speed(start_time, end_time, distance_covered_in_pixels)
                print(f"Current Speed: {current_speed} km/h")  # Display speed
                raise_alert(speed_threshold, current_speed)  # Check and raise alert
    
    def detect_accidents(self):
        if self.video_capture:
            frame_count = 0
            prev_frame = None
            alert_threshold = 100  # Adjust this threshold based on your scene

            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
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
                    self.alert_and_capture(frame)

                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
    def alert_and_capture(self, frame):
        print("ALERT: Accident Detected!")
        cv2.imwrite('accident_frame.jpg', frame)

    def detect_weapons(self):
        if self.video_capture:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                height, width, channels = frame.shape

                # Detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                self.model.setInput(blob)

                layer_names = self.model.getLayerNames()
                output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
                colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
                outs = self.model.forward(output_layers)

                # Showing information on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                if indexes is not None and len(indexes) > 0:
                    for i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        if label == "Weapon":
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

# Create the main window
root = tk.Tk()
app = VideoPlayerApp(root)
root.mainloop()