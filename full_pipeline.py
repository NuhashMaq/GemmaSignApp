import cv2
import mediapipe as mp
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3

# Load your trained gesture classifier
gesture_model = joblib.load("gesture_model.pkl")

# Initialize Gemma 3n tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def gemma_translate(gesture_label):
    prompt = f"Translate this sign language gesture to natural language: {gesture_label}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_prediction = None  # To avoid repeating speech too often
cooldown_frames = 0    # Simple cooldown counter

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Predict gesture label & confidence
            prediction = gesture_model.predict([landmarks])[0]
            confidence = max(gesture_model.predict_proba([landmarks])[0])

            cv2.putText(frame, f"{prediction} ({confidence:.2f})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Only speak if confidence > 0.8 and cooldown passed
            if confidence > 0.8:
                if prediction != last_prediction or cooldown_frames == 0:
                    translation = gemma_translate(prediction)
                    cv2.putText(frame, f"Gemma says: {translation}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    print(f"Gesture: {prediction} → Translation: {translation}")
                    speak(translation)
                    last_prediction = prediction
                    cooldown_frames = 50  # roughly 2 seconds cooldown (assuming ~25fps)

    # Reduce cooldown counter
    if cooldown_frames > 0:
        cooldown_frames -= 1

    cv2.imshow("Sign Language Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
