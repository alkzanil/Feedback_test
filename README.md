# Mental Health Application for Early Detection and Real-Time Feedback Using Multi-Modal Data 

This project focuses on developing an AI-driven application that leverages multi-modal data—facial expressions and speech features—to detect emotional states and provide real-time mental health feedback. The system integrates advanced machine learning models, specifically XGBoost and RFCNN, for emotion classification and feedback generation.

## Features

* Multi-Modal Data Fusion: Combines facial and speech features for enhanced emotion detection. 
* Real-Time Analysis: Processes user-uploaded videos to provide feedback within seconds.
* Feedback Generator: Offers emotion-specific suggestions, enhancing user engagement and mental health awareness.
* User-Friendly Interface: Built with Streamlit for a seamless video upload and analysis experience.
* Privacy-Focused: Ensures no permanent storage of uploaded videos and complies with data protection guidelines.

## Datasets

* CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset): Used for training the emotion detection model with diverse facial and speech data.

## Methodology

* Preprocessing: Extracts facial features using DeepFace and speech features using MFCC.
* Model Training: Employs XGBoost and RFCNN for multi-modal emotion classification.
* Real-Time Feedback System: Delivers feedback and confidence scores based on detected emotions.
* Streamlit Web App: Facilitates easy user interaction and real-time emotion analysis.

## Results

* XGBoost Accuracy: Achieved an accuracy of 74% during testing.
* Feedback Generator Performance: High-confidence results for emotions like happiness and calmness, with actionable suggestions for user well-being.

## Future Scope

* Incorporate additional modalities (e.g., text or EEG data) for deeper analysis.
* Expand the system for long-term mental health monitoring.
* Collaborate with healthcare providers for clinical integration.

## Installation and Usage

* Clone this repository:  git clone https://github.com/your-repo-name.git
* Install dependencies:
pip install -r requirements.txt
* Run the application:
streamlit run main.py
