import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import joblib
from PIL import Image

# Load the trained model
clf = joblib.load('modelok.pkl')

# Initialize MediaPipe Pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to extract landmarks
def extract_landmarks_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        points = [
            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
        ]
        return np.array(points).flatten()
    return None

# Streamlit app setup
st.title("Real-Time Pose Detection System")

# Upload video option
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded file to disk to process with OpenCV
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    # Read video file
    cap = cv2.VideoCapture("uploaded_video.mp4")

    # Loop through frames
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract landmarks from each frame
        landmarks = extract_landmarks_from_frame(frame)
        if landmarks is not None:
            # Reshape and predict using the classifier
            feature_vector = landmarks.reshape(1, -1)
            prediction = clf.predict(feature_vector)
            prediction_prob = clf.predict_proba(feature_vector)

            # Get probabilities for each class
            prob_faint = prediction_prob[0][1] * 100
            prob_sitting = prediction_prob[0][2] * 100
            prob_standing = prediction_prob[0][3] * 100

            # Display prediction probabilities
            stframe.text(f"Faint Probability: {prob_faint:.2f}%")
            stframe.text(f"Sitting Probability: {prob_sitting:.2f}%")
            stframe.text(f"Standing Probability: {prob_standing:.2f}%")

            # Overlay text based on prediction
            if prob_faint > 90:
                cv2.putText(frame, 'Faint Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif prediction == 2:
                cv2.putText(frame, 'Sitting Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif prediction == 3:
                cv2.putText(frame, 'Standing Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display landmarks
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Convert frame to display in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

    cap.release()

st.write("Upload a video to see real-time pose detection probabilities.")
