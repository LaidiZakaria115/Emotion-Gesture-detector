import cv2
import mediapipe as mp
from fer import FER
import time

# Initialize MediaPipe and FER
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

face_detector = FER(mtcnn=True)

# Start camera
cap = cv2.VideoCapture(0)
p_time = 0

def get_gesture(landmarks):
    """Simple open/fist detection based on hand landmarks"""
    if not landmarks:
        return "No Hand"
    # Get y-coordinates of fingertips and MCP joints
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    open_fingers = 0
    for tip, mcp in zip(finger_tips, finger_mcp):
        if landmarks.landmark[tip].y < landmarks.landmark[mcp].y:
            open_fingers += 1

    if open_fingers == 0:
        return "Fist ✊"
    elif open_fingers == 5 or open_fingers >= 4:
        return "Open Hand ✋"
    elif open_fingers == 2:
        return "Peace ✌️"
    else:
        return f"{open_fingers} Fingers"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # HAND detection
    results = hands.process(frame_rgb)
    gesture = "No Hand"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks)

    # FACE emotion detection
    emotion_text = "No Face"
    try:
        emotion = face_detector.detect_emotions(frame)
        if emotion:
            top_emotion, score = max(emotion[0]["emotions"].items(), key=lambda x: x[1])
            emotion_text = f"{top_emotion.capitalize()} ({score*100:.1f}%)"
    except Exception as e:
        emotion_text = "Error detecting face"

    # FPS calculation
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) != 0 else 0
    p_time = c_time

    # Overlay info
    cv2.rectangle(frame, (10, 10), (400, 130), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Emotion: {emotion_text}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)

    cv2.imshow("Emotion + Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
