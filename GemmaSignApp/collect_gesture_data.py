import cv2
import mediapipe as mp
import csv
import os
import time

GESTURE_LABEL = "ILOVEYOU"
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, f"{GESTURE_LABEL}.csv")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

print(f"Collecting gesture data for: {GESTURE_LABEL}")
print("Starting in 3 seconds...")
time.sleep(3)

with open(csv_path, "w", newline="") as f:
    csv_writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Flatten all (x, y, z) into a list
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                # Save to CSV
                csv_writer.writerow(row)

        # Show camera feed
        cv2.imshow("Recording Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Data saved to: {csv_path}")
