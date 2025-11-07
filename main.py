import cv2
import mediapipe as mp
from fer import FER
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(False, 1, 0.7, 0.7)
face_detector = FER(mtcnn=True)

cap = cv2.VideoCapture(0)
p_time = 0

def get_gesture(landmarks):
    if not landmarks:
        return "No Hand"
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    open_fingers = sum(landmarks.landmark[tip].y < landmarks.landmark[mcp].y for tip, mcp in zip(finger_tips, finger_mcp))
    if open_fingers == 0:
        return "✊"
    elif open_fingers >= 4:
        return "✋"
    elif open_fingers == 2:
        return "✌️"
    return f"{open_fingers} Fingers"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    gesture = "No Hand"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks)

    emotion_text = "No Face"
    try:
        emotions = face_detector.detect_emotions(frame)
        if emotions:
            top, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
            emotion_text = f"{top.capitalize()} ({score*100:.0f}%)"
    except:
        emotion_text = "Face Error"

    c_time = time.time()
    fps = 1 / (c_time - p_time + 1e-6)
    p_time = c_time

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, f"FPS: {int(fps)}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture}", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Emotion: {emotion_text}", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)

    cv2.imshow("Emotion + Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
