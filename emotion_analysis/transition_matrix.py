import re
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize a list to store emotions in order
emotion_transitions = []

# Read the log file
log_file_path = r"Transcriptions&Emotions.log"  # Log file with transcriptions and respective emotions
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# Define a regex pattern to extract timestamp, emotions, and confidence
pattern = re.compile(r"(\d{4}-\d{2}-\d{2}, \d{2}:\d{2}:\d{2}) - Transcription: .* Emotion: (\w+). Confidence: ([0-9.]+)")

# Extract timestamp, emotion, and confidence from the log lines
for line in lines:
    match = pattern.search(line)
    if match:
        timestamp = match.group(1)
        emotion = match.group(2)
        confidence = float(match.group(3))
        
        # Append the emotion to the transitions list
        emotion_transitions.append(emotion)

# Create the transition matrix
transitions = defaultdict(Counter)
for (e1, e2) in zip(emotion_transitions, emotion_transitions[1:]):
    transitions[e1][e2] += 1

# Convert to a DataFrame for easier manipulation
transition_df = pd.DataFrame(transitions).fillna(0)

# Normalize the transition counts by row to get probabilities
transition_prob_df = transition_df.div(transition_df.sum(axis=1), axis=0)

# Plotting the transition matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(transition_prob_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('Emotion Transition Matrix')
plt.xlabel('Next Emotion')
plt.ylabel('Previous Emotion')
plt.show()
