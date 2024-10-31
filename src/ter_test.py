import speech_recognition as sr
from transformers import pipeline
import logging
import matplotlib.pyplot as plt
from collections import Counter
import threading
import time

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize the emotion classification pipeline with EmoRoBERTa Model
pipe = pipeline("text-classification", model='arpanghoshal/EmoRoBERTa')

# Configure logging to include time and date
logging.basicConfig(
    filename='transcriptions_emotions2.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d, %H:%M:%S'
)

# For real-time visualization
emotions_counter = Counter()

# Flag to stop the loop
stop_listening = False

def listen_and_classify():
    global stop_listening
    with sr.Microphone() as source:
        print("Listening for speech...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcribed text: {text}")

        # Check for the stopping word
        if "goodbye" in text.lower():
            print("Stopping word detected. Ending conversation.")
            stop_listening = True
            return

        emotion_labels = pipe(text)
        main_emotion = max(emotion_labels, key=lambda x: x['score'])
        emotion = main_emotion['label']
        confidence = main_emotion['score']
        print(f"Emotion: {emotion}, Confidence: {confidence:.2f}")

        # Log transcription, emotion, and confidence with time and date
        logging.info(f"Transcription: {text}. Emotion: {emotion}. Confidence: {confidence:.2f}")

        # Update emotions counter
        emotions_counter[emotion] += 1

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def continuous_listen_and_classify():
    while not stop_listening:
        listen_and_classify()
        time.sleep(1)  # Add a short delay to avoid overwhelming the system

# Start the continuous listening and classification
continuous_listen_and_classify()
