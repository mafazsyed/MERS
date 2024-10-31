# Multimodal Emotion and Physiological State Recognition for Humanoid Robots

This repository contains the code and resources developed as part of the thesis project *Multimodal Emotion and Physiological State Recognition for Humanoid Robots*. The project aims to advance human-robot interaction (HRI) through real-time emotion and physiological state recognition using multimodal data.

## Abstract

This thesis introduces a **Multimodal Emotion Recognition System (MERS)** for humanoid robots, combining **Facial, Text, and Speech Emotion Recognition** to enable robust, contactless emotion monitoring. By unifying emotion labels and applying weighted emotion aggregation, MERS achieves adaptability for resource-constrained, real-time applications.

### Key Contributions
- **Facial Emotion Recognition (FER)**: Using CNN and MobileNetFER13 models trained on the FER2013 dataset.
- **Text Emotion Recognition (TER)**: An LSTM-based model trained on GoEmotions for nuanced text emotion analysis.
- **Text Sentiment Analysis Integration**: Facilitates dynamic feedback in the JD Humanoid Robot based on detected user emotions.
- **Remote Photoplethysmography (rPPG)**: An Eulerian Video Magnification-based model estimates heart rate in a contactless manner.

While each model showed promising accuracy, further work is needed for underrepresented emotion classes and expanded dataset validation. Future development can enhance MERS's applicability in healthcare, education, and social robotics, potentially transforming human-robot interaction in real-time applications.

## Project Structure
- **/src**: Source code for FER, TER, and rPPG models.
- **/models**: Pre-trained model weights for FER, TER, and rPPG (FER CNN, MobileNetFER13, and LSTM for TER).
- **/docs**: Project documentation, including system design, model architecture, and evaluation.
- **/emotion_analysis**: Scripts for emotional analysis performed via logged transcriptions and emotions, with examples.
- **/notebooks**: Jupyter notebooks for model training and testing.
- **/robot_integration**: Scripts for deploying emotion recognition on the JD Humanoid Robot for real-time, personalized feedback.

## Training Models
Each developed model can be trained individually using the notebooks in `/notebooks`:

### Face Emotion Recognition
#### Baseline FER
The baseline FER CNN model was trained on the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). The model consists of five convolutional layers with increasing filter sizes (32, 64, 128, and 256), each followed by batch normalization for improved training stability. Max-pooling layers are applied after certain convolutional blocks to reduce spatial dimensions, and dropout layers are included to prevent overfitting. Two fully connected layers are used, with the final dense layer comprising seven neurons and a softmax activation function, corresponding to the seven emotion classes.

Model trained using `Baseline_FER_CNN.ipynb`.

#### MobileNet-FER13
The base architecture of MobileNet was used with pre-trained weights from ImageNet. The model was trained on the [cleaned version of the FER2013 dataset](https://www.kaggle.com/datasets/gauravsharma99/fer13-cleaned-dataset), which removed inconsistencies and included five emotion classes: 'Angry, Fear, Happy, Disgust, and Neutral'.

Model trained using `MobileNet-FER13.ipynb`.

### Text Emotion Recognition
#### LSTM-GoEmotions
For text emotion recognition, a Long Short-Term Memory (LSTM) model was chosen due to its ability to capture long-range dependencies in sequential data like text, while remaining computationally efficient compared to more complex models like BERT. The model was trained using the [GoEmotions dataset](https://www.kaggle.com/datasets/debarshichanda/goemotions).

Model trained using `TER_LSTM.ipynb`.

## Pre-Trained Models

### Multimodal Emotion Recognition System (MERS)
MERS combines pre-trained FER, TER, and SER models—[DeepFace](https://pypi.org/project/deepface/0.0.24/), [EkmanClassifier](https://huggingface.co/arpanghoshal/EkmanClassifier), and [Hubert-Large](https://huggingface.co/superb/hubert-large-superb-er), respectively—using a late fusion model. The late fusion model uses a weighted emotion and modal aggregation method to produce a final `aggregated_emotion`.

![fusion_model_mers](https://github.com/user-attachments/assets/3eabc7a8-ccfe-48c2-9b41-31f8fcb2f992)

#### Combined Emotion Aggregation Equation

The combined emotion score \( E_i \) is calculated as:

![CodeCogsEqn](https://github.com/user-attachments/assets/ce65b40a-c056-4818-8fa0-cda86d77bc5e)

where:

![CodeCogsEqn (1)](https://github.com/user-attachments/assets/5a94e679-9fd0-406e-84fc-221b0b4f5cae)

![CodeCogsEqn (2)](https://github.com/user-attachments/assets/60991a6d-fb06-40d9-9620-d610021ce128)

##### Parameters
- \( E_i \): Combined score for the \( i \)th emotion.
- \( F_i \), \( T_i \), and \( S_i \): Emotion scores from face, text, and speech recognition models for the \( i \)th emotion.
- \( w_{\text{face}}, w_{\text{text}}, w_{\text{speech}} \): Weights assigned to the face, text, and speech modalities.
- \( n_i \): Emotion-specific multiplier adjusting the influence of the speech emotion score to map the four emotion classes in speech recognition to the seven classes used in face and text models.




![combined_emotion_flowchart](https://github.com/user-attachments/assets/1fcfe54d-5ffe-4bde-8918-08e8075ff11d)

  
  Run `MERS_Final.py` to observe multimodal emotion analysis in action. Emotions are logged to a `.log` file, enabling further emotional analysis.

- **Emotional Analysis**: The `emotion_analysis` folder contains scripts and examples for emotional analysis, including:
  - Emotion distribution graphs (bar and pie charts)
  - Temporal analysis (time-based analysis of emotions)
  - Emotion transition matrix
  
### Remote Photoplethysmography (rPPG)
This project includes an enhanced version of an existing [rPPG model](https://github.com/giladoved/webcam-heart-rate-monitor) based on [Eulerian Video Magnification](https://people.csail.mit.edu/mrub/papers/vidmag.pdf). Run it via `rppg_heartrate_enhanced.py` to estimate heart rate.

![rppg_flowchart](https://github.com/user-attachments/assets/6056d2e4-a6da-468b-aa85-4b9832e28f2c)

| Enhancement                     | Description                                                                                                 |
|---------------------------------|-------------------------------------------------------------------------------------------------------------|
| Face Detection                  | Uses OpenCV’s Haar Cascade for dynamic face region detection instead of a fixed window.                    |
| Webcam Resolution & Frame Rate  | Set to 640x480 pixels at 30 fps for higher data accuracy.                                                  |
| Gaussian Pyramid Construction   | Reduced from 3 to 2 levels for faster processing; face region resized to 160x120 pixels.                   |
| Heart Rate Calculation Frequency| Calculated every 30 frames with a 20 BPM buffer to smooth fluctuations.                                    |
| Bandpass Filtering              | 1.0–2.0 Hz filter isolates heart rate (60–120 BPM) while reducing noise and motion artefacts.              |
| Real-Time Visualization         | Displays detected face ROI and heart rate in real-time with historical annotations.                        |

### Deploying on JD Humanoid Robot

The `jd_humanoid_integration.EZB` file integrates:
- **Text Sentiment Analysis** using the VADER Sentiment Analysis module.
- **Speech Recognition and Speech-to-Text** via the Bing Speech Recognition module, with auto wake-up functionality.
- **Visual Input Analysis** from the robot’s camera using the Microsoft Cognitive Vision module.
- **OpenAI GPT-4 Module** to generate tailored responses based on speech, vision, and sentiment.

Other modules, such as RGB Animator, were used for enhanced and further tailored feedback based on text sentiment analysis results.

![jd_integration](https://github.com/user-attachments/assets/37b6bb19-ca14-4310-8fee-94028438fc7b)

## Future Work

- **Emotion Recognition**:
  - Speech diarization, face recognition, and Voice Activity Detection (VAD).
  - Expanding/merging datasets for diverse emotions and demographics.
  - Evaluating MERS on datasets like Multimodal EmotionLines Dataset (MELD) or Interactive Emotional Dyadic Motion Capture (IEMOCAP).
  - Developing and comparing other fusion models, such as early, hybrid, and intermediate fusion, with the late fusion model.
  - Exploring advanced fusion techniques like deep neural networks, ensemble methods, or attention mechanisms for fine-tuning weights.
  - Data augmentation, hyperparameter tuning, and trying other pre-trained models like VGG-16 or ResNet.
  
- **rPPG Heart Rate Detection**:
  - Validating the model using all subjects of the UBFC-rPPG dataset.
  - Face skin segmentation, such as Skin-SegNet, instead of traditional face detection; use of deep learning-based rPPG methods.
