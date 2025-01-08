import os
import numpy as np
import pandas as pd
import xgboost as xgb
from moviepy.editor import VideoFileClip
import librosa
import joblib
import cv2
import streamlit as st
from deepface import DeepFace

# Load the trained model
model_path = "C:\\Users\\alkz0\\Downloads\\dessertation\\xgboost_model.pkl"  # Adjust the path as needed
xgb_model = joblib.load(model_path)

# Emotion mapping for facial features
emotion_mapping = {
    'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'SAD': 4, 'NEU': 5
}

# Extract facial features
def extract_facial_features(video_frames, frame_interval=30):
    facial_features = []
    for i, frame in enumerate(video_frames):
        if i % frame_interval == 0:
            # Save the frame temporarily
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            # Use DeepFace to extract emotion scores
            try:
                analysis = DeepFace.analyze(img_path=temp_frame_path, actions=['emotion'], enforce_detection=False)[0]
                emotion_scores = analysis['emotion']
                # Map and flatten scores
                flattened_emotion_scores = [emotion_scores.get(emotion.lower(), 0) for emotion in emotion_mapping.keys()]
                facial_features.append(flattened_emotion_scores)
            except Exception as e:
                facial_features.append([0] * len(emotion_mapping))  # Zero padding for failed frames

    if not facial_features:
        raise ValueError("No facial features extracted.")

    # Average features across processed frames
    return np.mean(facial_features, axis=0)

# Extract audio features
def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfccs.mean(axis=1).tolist()
    except Exception as e:
        raise ValueError(f"Error extracting audio features: {e}")

# Process video and audio
def process_video(video_path):
    try:
        clip = VideoFileClip(video_path)
        video_frames = list(clip.iter_frames(fps=5))  # Match FPS used during preprocessing
        audio_path = video_path.replace('.mp4', '.wav')
        clip.audio.write_audiofile(audio_path)

        clip.close()  # Ensure the clip is released

        # Extract features
        facial_features = extract_facial_features(video_frames)
        speech_features = extract_audio_features(audio_path)

        # Concatenate features
        input_features = np.hstack([facial_features, speech_features])

        # Verify feature length
        expected_length = xgb_model.n_features_in_
        if len(input_features) != expected_length:
            raise ValueError(f"Feature length mismatch. Expected: {expected_length}, Got: {len(input_features)}")

        return input_features

    except Exception as e:
        raise ValueError(f"Error processing video: {e}")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Feedback system
def feedback_system(video_path):
    input_features = process_video(video_path)
    emotion_label = xgb_model.predict([input_features])[0]

    emotion_mapping = {
        0: "Irritability Detected",
        1: "Annoyed",
        2: "High Anxiety",
        3: "Healthy",
        4: "Depression Warning",
        5: "Emotionally Stable"
    }
    feedback = emotion_mapping.get(emotion_label, "Unknown")

    suggestion_mapping = {
        0: "Hi! Let's take a deep breath and count to 10.... Feel better?",
        1: "Hi! You seem annoyed, let's hear your favourite song!",
        2: "Hi! I feel you are anxious, no need to worry, you are not alone!",
        3: "Hi! I am thrilled that you are healthy and happy!",
        4: "Hi...I am sorry you feel that way, we will work this out, now let's take a deep breath.",
        5: "Hi! Well let's keep going!"
    }
    suggestion = suggestion_mapping.get(emotion_label, "No suggestion available.")

    return feedback, suggestion

# Streamlit App
st.title("Mental Health Feedback System")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "flv"])

if uploaded_file is not None:
    temp_video_path = "uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        feedback, suggestion = feedback_system(temp_video_path)
        st.success(f"Feedback: {feedback}")
        st.info(f"Suggestion: {suggestion}")
    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
