import re
from collections import Counter
import matplotlib.pyplot as plt

# Initialize the emotions counter
emotions_counter = Counter()

# Read the log file
log_file_path = r"Transcriptions&Emotions.log"  # Log file with transcriptions and respective emotions
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# Define a regex pattern to extract emotions
pattern = re.compile(r"Emotion: (\w+). Confidence: ([0-9.]+)")

# Extract emotions from the log lines
for line in lines:
    match = pattern.search(line)
    if match:
        emotion = match.group(1)
        confidence = float(match.group(2))
        # Increment the emotion count
        emotions_counter[emotion] += 1

# Print the counts for verification
print(emotions_counter)

# Plotting the emotion distribution
emotions = list(emotions_counter.keys())
counts = list(emotions_counter.values())

plt.figure(figsize=(10, 6))
plt.bar(emotions, counts, color='skyblue')
plt.xlabel('Emotions')
plt.ylabel('Counts')
plt.title('Emotion Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(counts, labels=emotions, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title('Emotion Distribution')
plt.show()
