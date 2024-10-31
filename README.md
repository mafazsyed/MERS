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
/src # Source code for FER, TER, and rPPG models.
/models # Pre-trained model weights for FER, TER, and rPPG (FER CNN, MobileNetFER13, and LSTM for TER).
/docs # Project documentation, including system design, model architecture, and evaluation.
/emotion_analysis # Scripts for emotional analysis performed via logged transcriptions and emotions, with examples.
/notebooks # Jupyter notebooks for model training and testing.
/robot_integration # Scripts for deploying emotion recognition on the JD Humanoid Robot for real-time, personalized feedback.

## Usage

### Training Models
Each developed model can be trained individually using the notebooks in `/notebooks`:

- **FER**: Train using `Baseline_FER_CNN.ipynb` and `MobileNet-FER13.ipynb`.
- **TER**: Train using `TER_LSTM.ipynb`.

### Pre-Trained Models

- **Multimodal Emotion Recognition System (MERS)**: MERS combines pre-trained FER, TER, and SER models—DeepFace, EkmanClassifier, and Hubert-Large, respectively—using a late fusion model. The late fusion model uses a weighted emotion and modal aggregation method to produce a final `aggregated_emotion`.

  ![MERS Model Diagram](/path/to/image)
  
  Run `MERS_Final.py` to observe multimodal emotion analysis in action. Emotions are logged to a `.log` file, enabling further emotional analysis.

- **Emotional Analysis**: The `emotion_analysis` folder contains scripts and examples for emotional analysis, including:
  - Emotion distribution graphs (bar and pie charts)
  - Temporal analysis (time-based analysis of emotions)
  - Emotion transition matrix
  
- **Remote Photoplethysmography (rPPG)**: This project includes an enhanced version of an existing rPPG model based on Eulerian Video Magnification. Run it via `rppg_heartrate_enhanced.py` to estimate heart rate. 

  | Enhancement | Description |
  | ----------- | ----------- |
  | [Add enhancement 1] | Description 1 |
  | [Add enhancement 2] | Description 2 |

### Deploying on JD Humanoid Robot

The `jd_humanoid_integration.EZB` file integrates:
- **Text Sentiment Analysis** using the VADER Sentiment Analysis module.
- **Speech Recognition and Speech-to-Text** via the Bing Speech Recognition module, with auto wake-up functionality.
- **Visual Input Analysis** from the robot’s camera using the Microsoft Cognitive Vision module.
- **OpenAI GPT-4 Module** to generate tailored responses based on speech, vision, and sentiment.
- **RGB Animator and Other Modules** for enhanced, tailored feedback based on text sentiment analysis.

![Process Flowchart](/path/to/flowchart)

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
