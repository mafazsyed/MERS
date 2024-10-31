import cv2
from transformers import pipeline
from deepface import DeepFace
from huggingface_hub import login
import threading
import queue
import speech_recognition as sr
import time
from collections import Counter
from matplotlib.patches import Wedge
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import torch
import torchaudio

# Login to Hugging Face hub
login(token="HuggingFace_Login_Token")

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Emotion categories and corresponding colors
emotions_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
colors = {
    'angry': 'red', 'disgust': 'green', 'fear': 'purple', 
    'happy': 'yellow', 'sad': 'blue', 'surprise': 'orange', 'neutral': 'gray'
}

# Initialize the text emotion classification pipeline
text_emotion_model = pipeline("text-classification", model='arpanghoshal/EkmanClassifier', top_k=None) # EkmanClassifier Model

# Create the pipeline for audio classification
audio_pipeline = pipeline("audio-classification", model="superb/hubert-large-superb-er") # Hubert-Large Model

# For real-time visualization
emotions_counter = Counter()

# Flag to stop the loop
stop_listening = False

# Queue for transferring text and speech emotions between threads
text_emotion_queue = queue.Queue()
speech_emotion_queue = queue.Queue()

# Initialize plot
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()

# Hide axes
ax.axis('off')

def listen_and_classify():
    global stop_listening
    with sr.Microphone() as source:
        print("Listening for speech...")
        audio = recognizer.listen(source, timeout=100, phrase_time_limit=10)
        audio_data = audio.get_wav_data()

    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcribed text: {text}")

        # Check for the stopping word
        if "goodbye" in text.lower():
            print("Stopping word detected. Ending conversation.")
            stop_listening = True
            return

        # Text emotion classification
        emotion_labels = text_emotion_model(text)

        # Flatten the list if necessary
        if isinstance(emotion_labels, list) and len(emotion_labels) > 0 and isinstance(emotion_labels[0], list):
            emotion_labels = emotion_labels[0]  

        # Print all emotion scores
        for emotion in emotion_labels:
            main_text = max(emotion_labels, key=lambda x: x['score'])
            main_text_emotion = main_text['label']
            main_text_emotion_score = main_text['score']

            text_emotion = emotion['label']
            text_confidence = emotion['score']
            # print(f"Text Emotion: {text_emotion}, Confidence: {text_confidence:.2f}")
            
            # Update emotions counter
            emotions_counter[text_emotion] += 1

        # print(f"Dominant Text Emotion: {main_text_emotion}, Confidence: {main_text_emotion_score:.2f}")
        
        # Put the latest text emotion into the queue
        text_emotion_queue.put((main_text_emotion, main_text_emotion_score))

        # Save audio to a temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)

        # In the function where you handle the speech recognition and emotion classification
        speech_emotion_results = audio_pipeline("temp_audio.wav")

        # Print and log all emotion scores
        for emotion in speech_emotion_results:
            max_speech_emotion = speech_emotion_results[0]['label']
            max_speech_emotion_score = speech_emotion_results[0]['score']

            speech_emotion = emotion['label']
            speech_confidence = emotion['score']
            # print(f"Speech Emotion: {speech_emotion}, Confidence: {speech_confidence:.2f}")

            # Update emotions counter
            emotions_counter[speech_emotion] += 1

        # print(f"Dominant Speech Emotion: {max_speech_emotion}, Confidence: {max_speech_emotion_score:.2f}")
        
        # Put the latest speech emotion and confidence into the queue
        speech_emotion_queue.put((max_speech_emotion, max_speech_emotion_score))

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def continuous_listen_and_classify():
    while not stop_listening:
        listen_and_classify()
        time.sleep(1)  # Add a short delay to avoid overwhelming the system

# Start the speech recognition thread
speech_thread = threading.Thread(target=continuous_listen_and_classify)
speech_thread.start()

def analyze_face_emotion(face_roi, result_queue):
    try:
        face_analysis = DeepFace.analyze(img_path=face_roi, actions=['emotion'], enforce_detection=False) # DeepFace Model

        # Check if results are in a list and handle accordingly
        if isinstance(face_analysis, list):
            result = face_analysis[0]  # Assuming the first result relates to the detected face
        else:
            result = face_analysis

        all_face_emotions = {k: v / 100 for k, v in result['emotion'].items()}  # Normalize to 0-1 range

        emotion = result['dominant_emotion']
        emotion_confidence = all_face_emotions[emotion]

        # for key, value in all_face_emotions.items():
            # print(f"Face Emotion: {key}, Confidence: {value:.2f}")

        # Put the normalized dictionary of face emotions into the queue
        result_queue.put((all_face_emotions, emotion, emotion_confidence))

    except Exception as e:
        print(f"Error analyzing face emotion: {e}")
        result_queue.put(({}, "unknown", 0.0))  # Return an empty dictionary on error

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

frame_count = 0
result_queue = queue.Queue()

# Initialize emotion and confidence variables with default values
face_emotion = "unknown"
face_confidence = 0.0
main_text_emotion = "unknown"
main_text_emotion_score = 0.0
max_speech_emotion = "unknown"
max_speech_emotion_score = 0.0
all_face_emotions = {}  # Initialize all_face_emotions as an empty dictionary
text_confidence = [0] * 7  # Initialize text_confidence with 7 zeros
speech_confidence = [0] * 4  # Initialize speech_confidence with 4 zeros

frame_interval = 15  # Perform face emotion analysis every 15 frames
current_frame = 0
last_emotions = None

# Map the text emotion labels to the corresponding labels in `emotions_list`
text_emotion_mapping = {
    'anger': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'joy': 'happy',        # Assuming 'joy' corresponds to 'happy'
    'neutral': 'neutral',
    'sadness': 'sad',
    'surprise': 'surprise'
}

w_face = 0.45
w_text = 0.35
w_speech = 0.20

# Main loop for video capture and face emotion detection
while True:
    if stop_listening:
        break  # Exit the loop if stop_listening is True

    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1  # Increment frame count
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if current_frame % frame_interval == 0 or last_emotions is None:
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            threading.Thread(target=analyze_face_emotion, args=(face_roi, result_queue)).start()
    else:
        all_face_emotions = last_emotions  # Use last detected emotions

    # Fetch new emotion data if available
    if not result_queue.empty():
        all_face_emotions, face_emotion, face_confidence = result_queue.get()
        last_emotions = all_face_emotions  # Store last emotions for next frames

    if not text_emotion_queue.empty():
        main_text_emotion, main_text_emotion_score = text_emotion_queue.get()

        mapped_emotion = text_emotion_mapping.get(main_text_emotion, None)
        if mapped_emotion and mapped_emotion in emotions_list:
            text_confidence[emotions_list.index(mapped_emotion)] = main_text_emotion_score

    if not speech_emotion_queue.empty():
        max_speech_emotion, max_speech_emotion_score = speech_emotion_queue.get()
        speech_confidence[['neu', 'hap', 'ang', 'sad'].index(max_speech_emotion)] = max_speech_emotion_score

    # Calculate the combined emotions
    combined_emotions = [0] * 7

    # Anger = Anger (Face) + Anger (Text) + Anger (Speech)
    combined_emotions[0] = (w_face * all_face_emotions.get('angry', 0)) + (w_text * text_confidence[0]) + (w_speech * 0.8 * speech_confidence[2])

    # Disgust = Disgust (Face) + Disgust (Text) + 0.4 * Sad (Speech)
    combined_emotions[1] = (w_face * all_face_emotions.get('disgust', 0)) + (w_text * text_confidence[1]) + (w_speech * 0.4 * speech_confidence[3])

    # Fear = Fear (Face) + Fear (Text) + 0.4 * Anger (Speech)
    combined_emotions[2] = (w_face * all_face_emotions.get('fear', 0)) + (w_text * text_confidence[2]) + (w_speech * 0.4 * speech_confidence[2])

    # Happy = Happy (Face) + Joy (Text) + Happy (Speech)
    combined_emotions[3] = (w_face * all_face_emotions.get('happy', 0)) + (w_text * text_confidence[3]) + (w_speech * speech_confidence[1])

    # Neutral = Neutral (Face) + Neutral (Text) + Neutral (Speech)
    combined_emotions[4] = (w_face * all_face_emotions.get('neutral', 0)) + (w_text * text_confidence[4]) + (w_speech * speech_confidence[0])

    # Sadness = Sadness (Face) + Sad (Text) + Sad (Speech)
    combined_emotions[5] = (w_face * 0.75 * all_face_emotions.get('sad', 0)) + (w_text * text_confidence[5]) + (w_speech * 0.25 * speech_confidence[3])

    # Surprise = Surprise (Face) + Surprise (Text) + 0.2 * Anger (Speech)
    combined_emotions[6] = (w_face * all_face_emotions.get('surprise', 0)) + (w_text * text_confidence[6]) + (w_speech * 0.4 * speech_confidence[2])

    # Normalize the combined emotions so they sum up to 1 (or 100%)
    total = sum(combined_emotions)
    if total > 0:
        combined_emotions = [x / total for x in combined_emotions]

    # Find the max combined emotion and its label
    max_combined_emotion = max(combined_emotions)
    max_combined_emotion_label = emotions_list[combined_emotions.index(max_combined_emotion)]

    # Print the max combined emotion and its label
    # print(f"Max Combined Emotion: {max_combined_emotion_label} ({max_combined_emotion:.2f})")

    # Clear the previous plot
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')  # Hide axes again after clearing

    # Draw each wedge with the appropriate radial length and transparency based on the confidence score
    start_angle = 90
    for i, emotion in enumerate(emotions_list):
        conf = combined_emotions[i]
        radius = 1.0 * conf  # Scale radius based on confidence (0-1 range)
        color = colors[emotion]
        rgba_color = tuple(list(mcolors.to_rgba(color)[:3]) + [conf])  # Add alpha transparency based on confidence
        wedge = Wedge(center=(0, 0), r=radius, theta1=start_angle, theta2=start_angle + 360 / len(emotions_list), 
                        facecolor=rgba_color, edgecolor='none')  # No outline
        ax.add_patch(wedge)

        # Calculate the position for the label at the edge of the slice
        mid_angle = start_angle + (360 / len(emotions_list)) / 2
        label_x = 1.3 * np.cos(np.radians(mid_angle))
        label_y = 1.3 * np.sin(np.radians(mid_angle))

        # Add the label with the emotion and confidence score
        ax.text(label_x, label_y, f'{emotion}\n{conf*100:.2f}%', ha='center', va='center', color='black', fontsize=10, fontweight='bold')

        start_angle += 360 / len(emotions_list)

    # Update the plot
    plt.draw()
    plt.pause(0.001)  # Pause to update the figure
    
    # Draw results on the frame for each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"Face: {face_emotion} ({face_confidence:.2f})", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {main_text_emotion} ({main_text_emotion_score:.2f})", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Speech: {max_speech_emotion} ({max_speech_emotion_score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Combined: {max_combined_emotion_label} ({max_combined_emotion:.2f})", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_listening = True
speech_thread.join()
plt.close()