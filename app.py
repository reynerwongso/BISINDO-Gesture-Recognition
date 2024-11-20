import cv2
import tensorflow
import numpy as np
import mediapipe as mp
from tensorflow import keras
from keras.models import load_model
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('FC.h5')

labelsAbjad = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'none', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '0',
    'Apa', 'Siapa', 'Kapan', 'Saya', 'Anda', 'Buku', 'Kacamata', 'Ayah', 'Ibu', 'Mobil', 'Motor', 'Belajar', 'Transportasi'
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image):
    """Extract hand landmarks from the image using MediaPipe"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (128, 128))
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        all_landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            all_landmarks.append(landmarks)

        if len(all_landmarks) == 1:
            all_landmarks.append([0] * 63)

        return all_landmarks[0] + all_landmarks[1], results
    return None, results

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        landmarks, results = extract_landmarks(frame)

        if landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            label = labelsAbjad[predicted_class]

            cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

# if __name__ == '__main__':
#   app.run(host='0.0.0.0', port=5000, debug=True)

# if __name__ == '__main__':
#    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
