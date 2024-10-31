import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the video file
video_path = r"Input_Video_File.mp4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Can use other codecs like 'XVID'
out = cv2.VideoWriter(r"Output_Video_File.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))) # Annotated output video file

# Emotion categories and corresponding colors
emotions_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
colors = {
    'angry': 'red', 'disgust': 'green', 'fear': 'purple', 
    'happy': 'yellow', 'sad': 'blue', 'surprise': 'orange', 'neutral': 'gray'
}

# Initialize plot
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()

# Hide axes
ax.axis('off')

# Define the frame interval for emotion detection (e.g., 2 seconds at 20 FPS = 40 frames)
frame_interval = 12
current_frame = 0
last_emotions = None

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is captured

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    for (fx, fy, fw, fh) in faces:  # Use fx, fy, fw, fh for face coordinates
        if current_frame % frame_interval == 0 or last_emotions is None:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[fy:fy + fh, fx:fx + fw]  # Use original frame for analysis to get color data

            # Perform emotion analysis on the face ROI
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Check if results are in a list and handle accordingly
            if isinstance(results, list):
                result = results[0]  # Assuming the first result relates to the detected face
            else:
                result = results

            # Get the emotions and their confidence scores
            last_emotions = result['emotion']

        # Use the last calculated emotions for the current frame
        emotions = last_emotions

        # Clear the previous plot
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')  # Hide axes again after clearing

        # Draw each wedge with the appropriate radial length and transparency based on the confidence score
        start_angle = 90
        for i, emotion in enumerate(emotions_list):
            conf = emotions.get(emotion, 0)
            radius = 1.0 * conf / 100.0  # Scale radius based on confidence
            color = colors[emotion]
            rgba_color = tuple(list(mcolors.to_rgba(color)[:3]) + [conf / 100.0])  # Add alpha transparency based on confidence
            wedge = Wedge(center=(0, 0), r=radius, theta1=start_angle, theta2=start_angle + 360 / len(emotions_list), 
                          facecolor=rgba_color, edgecolor='none')  # No outline
            ax.add_patch(wedge)

            # Calculate the position for the label at the edge of the slice
            mid_angle = start_angle + (360 / len(emotions_list)) / 2
            label_x = 1.3 * np.cos(np.radians(mid_angle))
            label_y = 1.3 * np.sin(np.radians(mid_angle))

            # Add the label with the emotion and confidence score
            ax.text(label_x, label_y, f'{emotion}\n{conf:.2f}%', ha='center', va='center', color='black', fontsize=10, fontweight='bold')

            start_angle += 360 / len(emotions_list)

        # Draw rectangle around face
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)

        # Display the dominant emotion above the face
        dominant_emotion = max(emotions, key=emotions.get)
        cv2.putText(frame, f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}%", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with annotations to the output video
    out.write(frame)

    # Display the resulting frame (Optional, remove if you don't need to display)
    cv2.imshow('Real-time Emotion Detection', frame)

    # Increment the frame count
    current_frame += 1

    # Press 'q' to exit (Optional, remove if you don't need real-time display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
plt.close()
