import re
from collections import defaultdict, Counter  # Import Counter here
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Initialize a dictionary to store emotions by hour
emotions_by_hour = defaultdict(lambda: Counter())

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
        
        # Parse the hour from the timestamp
        hour = pd.to_datetime(timestamp).hour
        
        # Increment the emotion count for the corresponding hour
        emotions_by_hour[hour][emotion] += 1

# Convert to a DataFrame for easier manipulation
data = []
for hour, counter in emotions_by_hour.items():
    for emotion, count in counter.items():
        data.append({'hour': hour, 'emotion': emotion, 'count': count})

df = pd.DataFrame(data)

# Print the DataFrame for verification
print(df)

# Pivot the DataFrame for heatmap visualization
pivot_table = df.pivot_table(index='hour', columns='emotion', values='count', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Temporal Distribution of Emotions')
plt.xlabel('Emotion')
plt.ylabel('Hour of Day')
plt.show()
