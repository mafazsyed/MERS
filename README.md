# Multimodal Emotion and Physiological State Recognition for Humanoid Robots

This repository contains the code and resources developed as part of the thesis project *Multimodal Emotion and Physiological State Recognition for Humanoid Robots*. The project aims to advance human-robot interaction (HRI) through real-time emotion and physiological state recognition using multimodal data.

## Abstract
This project develops an emotion recognition system that leverages facial expressions, speech inputs, and textual content alongside remotely detecting physiological indicators like heart rate through contactless facial camera footage. Through the development, training, and testing of Facial Emotion Recognition (FER) models and Text Emotion Recognition (TER) techniques using established datasets, the project seeks to enhance and compare the accuracy and generalization capabilities of these models. Part of the emotion recognition system is integrated into the JD Humanoid Robot to demonstrate its practical applications in enhancing human-robot interaction.

### Video Preview
<div style="display: flex; justify-content: space-around;">

  <a href="https://mafazsyed.com/wp-content/uploads/2024/10/MERS-FER-Interstellar-Scene.mp4">
    <img src="https://github.com/user-attachments/assets/743fe3f5-3021-4fd3-85ae-159b96d60668" width="48%" />
  </a>

  <a href="https://mafazsyed.com/wp-content/uploads/2024/10/rPPG-Subject-3-UBFC-rPPG-Dataset-2.mp4">
    <img src="https://github.com/user-attachments/assets/893d3202-f4d3-49a2-a297-4831f37cdd9e" width="48%" />
  </a>

</div>

### Key Contributions
- **Facial Emotion Recognition (FER)**: Using CNN and MobileNetFER13 models trained on the FER2013 dataset.
- **Text Emotion Recognition (TER)**: An LSTM-based model trained on GoEmotions for nuanced text emotion analysis.
- **Text Sentiment Analysis Integration**: Facilitates dynamic feedback in the JD Humanoid Robot based on detected user emotions.
- **Multimodal Emotion Recognition System (MERS)**: Aggregation of facial, text, and speech recognition via a late fusion model to enable real-time, contactless emotion monitoring with emotion logging, visualisation, and analysis.
- **Remote Photoplethysmography (rPPG)**: An Eulerian Video Magnification-based model estimates heart rate in a contactless manner.

While each model showed promising accuracy, further work is needed for underrepresented emotion classes and expanded dataset validation. Future development can enhance MERS's applicability in healthcare, education, and social robotics, potentially transforming human-robot interaction in real-time applications.

## Repository Structure
- **/src**: Source code for FER, TER, and rPPG models.
- **/model_weights**: Pre-trained model weights for FER, TER, and rPPG (FER CNN, MobileNetFER13, and LSTM for TER).
- **/emotion_analysis**: Scripts for emotional analysis performed via logged transcriptions and emotions, with examples.
- **/notebooks**: Jupyter notebooks for model training and testing.
- **/robot_integration**: Scripts for deploying emotion recognition on the JD Humanoid Robot for real-time, personalized feedback.
- **/results**: Training progress, confusion matrices, and video results on samples.
- **/docs**: Project documentation, including system design, model architecture, and evaluation. *(Not included yet)*

## Training Models
Each developed model can be trained individually using the notebooks in `/notebooks`:

### Face Emotion Recognition
#### Baseline FER CNN Model
The baseline FER CNN model was trained on the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). The model consists of five convolutional layers with increasing filter sizes (32, 64, 128, and 256), each followed by batch normalization for improved training stability. Max-pooling layers are applied after certain convolutional blocks to reduce spatial dimensions, and dropout layers are included to prevent overfitting. Two fully connected layers are used, with the final dense layer comprising seven neurons and a softmax activation function, corresponding to the seven emotion classes.

Model trained using `Baseline_FER_CNN.ipynb`.

##### Results
![fer](https://github.com/user-attachments/assets/52fb3fc4-e60d-43dc-b04e-e102bb59f041)

| **Class**   | **Precision** | **Recall** | **F1-Score** | **Support** |
|-------------|---------------|------------|--------------|-------------|
| 0 (Angry)   | 0.58          | 0.57       | 0.58         | 958         |
| 1 (Disgust) | 0.69          | 0.41       | 0.52         | 111         |
| 2 (Fear)    | 0.52          | 0.36       | 0.43         | 1024        |
| 3 (Happy)   | 0.85          | 0.86       | 0.85         | 1774        |
| 4 (Sad)     | 0.53          | 0.48       | 0.50         | 1247        |
| 5 (Surprise)| 0.76          | 0.78       | 0.77         | 831         |
| 6 (Neutral) | 0.53          | 0.72       | 0.61         | 1233        |
| **Macro Avg**   | 0.64          | 0.60       | 0.61         | -           |
| **Weighted Avg**| 0.64          | 0.65       | 0.64         | -           |
| **Accuracy**    | -             | **0.65**   | -            | 7178        |

*Table: FER CNN model classification results*

#### MobileNet-FER13
The base architecture of MobileNet was used with pre-trained weights from ImageNet. The model was trained on the [cleaned version of the FER2013 dataset](https://www.kaggle.com/datasets/gauravsharma99/fer13-cleaned-dataset), which removed inconsistencies and included five emotion classes: 'Angry, Fear, Happy, Disgust, and Neutral'.

Model trained using `MobileNet-FER13.ipynb`.

##### Results
![mobilenet](https://github.com/user-attachments/assets/4eb8a3af-a34a-47a8-a0a2-4d7d58b97d96)

| **Class**   | **Precision** | **Recall** | **F1-Score** | **Support** |
|-------------|---------------|------------|--------------|-------------|
| Angry       | 0.60          | 0.64       | 0.62         | 283         |
| Fear        | 0.66          | 0.47       | 0.55         | 253         |
| Happy       | 0.89          | 0.87       | 0.88         | 684         |
| Neutral     | 0.65          | 0.76       | 0.70         | 430         |
| **Macro Avg**   | 0.70          | 0.68       | 0.69         | -           |
| **Weighted Avg**| 0.74          | 0.74       | 0.74         | -           |
| **Accuracy**    | -             | **0.74**   | -            | 1650        |

*Table: MobileNet-FER13 model classification results*


### Text Emotion Recognition
#### LSTM-GoEmotions
For text emotion recognition, a Long Short-Term Memory (LSTM) model was chosen due to its ability to capture long-range dependencies in sequential data like text, while remaining computationally efficient compared to more complex models like BERT. The model was trained using the [GoEmotions dataset](https://www.kaggle.com/datasets/debarshichanda/goemotions).

Model trained using `TER_LSTM.ipynb`.

##### Results

| **Class**      | **Precision** | **Recall** | **F1-Score** | **Support** |
|----------------|---------------|------------|--------------|-------------|
| 0 (Anger)      | 0.44          | 0.33       | 0.38         | 703         |
| 1 (Disgust)    | 0.94          | 0.94       | 0.94         | 2049        |
| 2 (Fear)       | 0.00          | 0.00       | 0.00         | 90          |
| 3 (Happy)      | 0.71          | 0.80       | 0.75         | 2049        |
| 4 (Sad)        | 0.60          | 0.37       | 0.45         | 317         |
| 5 (Surprise)   | 0.00          | 0.00       | 0.00         | 573         |
| 6 (Neutral)    | 0.48          | 0.69       | 0.57         | 1605        |
| **Macro Avg**      | 0.45          | 0.45       | 0.44         | -           |
| **Weighted Avg**   | 0.63          | 0.68       | 0.65         | -           |
| **Accuracy**       | -             | **0.68**   | -            | 7386        |

*Table: LSTM model classification results on GoEmotions dataset*


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

<img src="https://github.com/user-attachments/assets/d29567af-a0c5-45e8-86bb-3590314bfd9d" alt="weights" width="60%">

![combined_emotion_flowchart](https://github.com/user-attachments/assets/1fcfe54d-5ffe-4bde-8918-08e8075ff11d)

  
  Run `MERS_Final.py` to observe multimodal emotion analysis in action. Emotions are logged to a `.log` file, enabling further emotional analysis.

#### Result
![mers_preview](https://github.com/user-attachments/assets/f54e6584-2fe2-4757-95b6-0e86ac2020a2)

#### Emotional Analysis
The `emotion_analysis` folder contains scripts and examples for emotional analysis, including:
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

The model was tested on 'subject 3' of [UBFC-rPPG Dataset 2](https://sites.google.com/view/ybenezeth/ubfcrppg).

#### Model Performance on Subject 3
![rPPG Model Validation Graph](https://github.com/user-attachments/assets/62f0fd6d-f63b-4cc7-a79c-3903c7fe5c47)

### Integration with JD Humanoid Robot

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
