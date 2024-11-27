import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('FC.h5')

# Define labels
labelsAbjad = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'none', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '0',
    'Apa', 'Siapa', 'Kapan', 'Saya', 'Anda', 'Buku', 'Kacamata', 'Ayah', 'Ibu', 'Mobil', 'Motor', 'Belajar', 'Transportasi']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image):
    """Extract hand landmarks from the image using MediaPipe, ensuring output is always 2 * 63 keypoints if at least one hand is detected."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (128, 128))
    results = hands.process(image_rgb)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        all_landmarks = []

        # Iterate over each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            all_landmarks.append(landmarks)

        # If only one hand is detected, add a second set of zero landmarks
        if len(all_landmarks) == 1:
            all_landmarks.append([0] * 63)

        # Flatten and return the landmark list to ensure 2 * 63 keypoints
        return all_landmarks[0] + all_landmarks[1], results  # Return landmarks and results for visualization

    # Return None if no hands are detected
    return None, results  # Return None if no landmarks are detected

# Streamlit App
st.title("Real-Time Gesture Recognition with Hand Landmarks")

# Sidebar for settings
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Main App
st.write("Press 'Start Camera' to begin capturing gestures.")

# Initialize OpenCV webcam capture
if st.button("Start Camera"):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame.")
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Extract landmarks and get MediaPipe results for visualization
        landmarks, results = extract_landmarks(frame)

        if landmarks:
            # Use MediaPipe drawing utilities to draw landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to numpy array for model prediction
            input_data = np.array(landmarks).reshape(1, -1)  # Shape: (1, 126) for two hands (2 * 63)

            # Make prediction using the model
            prediction = model.predict(input_data)  # Assume 'model' is pre-loaded
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)  # Confidence score for the predicted class
            label = labelsAbjad[predicted_class]  # Map the class index to a label

            # Display prediction and confidence if it meets the threshold
            if confidence > confidence_threshold:
                cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame with landmarks and predictions
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

hands.close()
