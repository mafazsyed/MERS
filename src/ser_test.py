import speech_recognition as sr
from transformers import pipeline
import logging
import matplotlib.pyplot as plt
from collections import Counter
import threading
import time

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Create the pipeline for audio classification using the Hubert-Large model
audio_pipeline = pipeline("audio-classification", model="superb/hubert-large-superb-er")

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
        
        # Save the audio to a temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())

        # Get the emotion labels and confidence scores for the audio
        emotion_labels = audio_pipeline("temp_audio.wav")

        # Print and log all emotion scores
        for emotion in emotion_labels:
            label = emotion['label']
            confidence = emotion['score']
            print(f"Emotion: {label}, Confidence: {confidence:.2f}")
            logging.info(f"Emotion: {label}, Confidence: {confidence:.2f}")

            # Update emotions counter
            emotions_counter[label] += 1

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"Error processing audio: {e}")

def continuous_listen_and_classify():
    while not stop_listening:
        listen_and_classify()
        time.sleep(1)  # Add a short delay to avoid overwhelming the system

# Start the continuous listening and classification
continuous_listen_and_classify()
