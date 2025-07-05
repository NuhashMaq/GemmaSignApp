import cv2
import mediapipe as mp
import joblib
import numpy as np
import gradio as gr

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load your trained gesture recognition model
gesture_model = joblib.load("gesture_model.pkl")

def detect_gesture(video_frame):
    if video_frame is None:
        return "No frame", None

    img_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(img_rgb)

    if not hand_results.multi_hand_landmarks:
        return "No hand detected", video_frame

    hand_landmarks = hand_results.multi_hand_landmarks[0]

    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    landmarks_np = np.array(landmarks).reshape(1, -1)
    prediction = gesture_model.predict(landmarks_np)
    gesture_name = prediction[0]

    mp_drawing.draw_landmarks(video_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return gesture_name, video_frame

# Build the Gradio interface
iface = gr.Interface(
    fn=detect_gesture,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label="Detected Gesture"),
        gr.Image(label="Webcam Image with Landmarks")
    ],
    live=True,
    title="Sign Language Gesture Recognition"
)

# Launch with public link
if __name__ == "__main__":
    iface.launch(share=True)
