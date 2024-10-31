import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Helper Methods

# Builds a Gaussian pyramid for the input frame
def build_gaussian_pyramid(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

# Reconstructs a frame from the Gaussian pyramid
def reconstruct_frame(pyramid, index, levels):
    reconstructed_frame = pyramid[index]
    for level in range(levels):
        reconstructed_frame = cv2.pyrUp(reconstructed_frame)
    # Ensure the frame size matches the original dimensions
    reconstructed_frame = reconstructed_frame[:frame_height, :frame_width]
    return reconstructed_frame

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam Parameters
webcam = cv2.VideoCapture(sys.argv[1]) if len(sys.argv) == 2 else cv2.VideoCapture(0)
original_width = 640
original_height = 480
frame_width = 160
frame_height = 120
num_channels = 3
frame_rate = 30
webcam.set(3, original_width)
webcam.set(4, original_height)

# Color Magnification Parameters
gaussian_levels = 2
amplification_factor = 170
min_heartbeat_frequency = 1.0
max_heartbeat_frequency = 2.0
buffer_size = 150
current_buffer_index = 0

# Output Display Parameters
font_type = cv2.FONT_HERSHEY_SIMPLEX
calculating_text_position = (20, 30)
bpm_text_position = (frame_width // 2 + 5, 30)
font_scale = 1
font_color = (255, 255, 255)
font_thickness = 2
rectangle_color = (0, 255, 0)
rectangle_thickness = 3

# Initialize Gaussian Pyramid
initial_frame = np.zeros((frame_height, frame_width, num_channels))
initial_gaussian = build_gaussian_pyramid(initial_frame, gaussian_levels + 1)[gaussian_levels]
gaussian_video_buffer = np.zeros((buffer_size, initial_gaussian.shape[0], initial_gaussian.shape[1], num_channels))
fourier_transform_average = np.zeros((buffer_size))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0 * frame_rate) * np.arange(buffer_size) / (1.0 * buffer_size)
bandpass_filter_mask = (frequencies >= min_heartbeat_frequency) & (frequencies <= max_heartbeat_frequency)

# Heart Rate Calculation Variables
bpm_calculation_interval = 30
bpm_buffer_index = 0
bpm_buffer_size = 20
bpm_buffer = np.zeros((bpm_buffer_size))

# Matplotlib Figure Initialization for Live Plotting
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
time_data, bpm_data = [], []
line_plot, = plt.plot([], [], 'r-', animated=True)

# Add gridlines and labels
ax.grid(True)
ax.set_xlabel('Time (frames)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title('Real-Time Heart Rate Monitoring')

# Initialize the plot limits
def init_plot():
    ax.set_xlim(0, 100)
    ax.set_ylim(40, 180)
    return line_plot,

# Update the plot with new data
def update_plot(frame):
    line_plot.set_data(time_data, bpm_data)
    return line_plot,

# Create an animation object for updating the plot
animation = FuncAnimation(fig, update_plot, frames=np.arange(0, 200), init_func=init_plot, blit=True)

frame_count = 0
while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Assuming we take the first detected face
        (x, y, w, h) = faces[0]

        # Define the ROI as the detected face
        face_region = frame[y:y+h, x:x+w]

        # Resize the face region to match the expected video size
        resized_face_region = cv2.resize(face_region, (frame_width, frame_height))

        # Construct Gaussian Pyramid
        gaussian_video_buffer[current_buffer_index] = build_gaussian_pyramid(resized_face_region, gaussian_levels + 1)[gaussian_levels]
        fourier_transform = np.fft.fft(gaussian_video_buffer, axis=0)

        # Apply the bandpass filter
        fourier_transform[~bandpass_filter_mask] = 0

        # Calculate the Heart Rate (BPM)
        if current_buffer_index % bpm_calculation_interval == 0:
            frame_count += 1
            for buf_idx in range(buffer_size):
                fourier_transform_average[buf_idx] = np.real(fourier_transform[buf_idx]).mean()
            dominant_frequency = frequencies[np.argmax(fourier_transform_average)]
            bpm = 60.0 * dominant_frequency
            bpm_buffer[bpm_buffer_index] = bpm
            bpm_buffer_index = (bpm_buffer_index + 1) % bpm_buffer_size

            # Update the live graph
            time_data.append(frame_count)
            bpm_data.append(bpm_buffer.mean())
            ax.set_xlim(max(0, frame_count - 100), frame_count + 1)
            line_plot.set_data(time_data, bpm_data)
            plt.pause(0.01)

        # Amplify the filtered signal
        amplified_signal = np.real(np.fft.ifft(fourier_transform, axis=0)) * amplification_factor

        # Reconstruct the resulting frame
        filtered_frame = reconstruct_frame(amplified_signal, current_buffer_index, gaussian_levels)
        output_frame = resized_face_region + filtered_frame
        output_frame = cv2.convertScaleAbs(output_frame)

        # Update the buffer index
        current_buffer_index = (current_buffer_index + 1) % buffer_size

        # Place the processed frame back into the original frame
        frame[y:y+h, x:x+w] = cv2.resize(output_frame, (w, h))

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, rectangle_thickness)

        # Display BPM if enough data has been collected
        if frame_count > bpm_buffer_size:
            cv2.putText(frame, f"BPM: {int(bpm_buffer.mean())}", bpm_text_position, font_type, font_scale, font_color, font_thickness)
        else:
            cv2.putText(frame, "Calculating BPM...", calculating_text_position, font_type, font_scale, font_color, font_thickness)

        # Display the frame with the heart rate data
        if len(sys.argv) != 2:
            cv2.imshow("Webcam Heart Rate Monitor", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()

# Turn off interactive mode and show the final plot
plt.ioff()
plt.show()
