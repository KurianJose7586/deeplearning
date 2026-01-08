import cv2
import torch
import time
from collections import deque, Counter

from utils.face_utils import detect_face, preprocess_face
from posture.posture_utils import analyze_posture
from emotion.predict import load_model, predict_emotion
from utils.feedback import generate_feedback

# ---------------- Load Emotion Model ----------------
load_model("emotion/weights.pth")

# ---------------- Temporal Smoothing ----------------
EMOTION_WINDOW = 15
POSTURE_WINDOW = 15

emotion_buffer = deque(maxlen=EMOTION_WINDOW)
posture_buffer = deque(maxlen=POSTURE_WINDOW)

def majority_vote(buffer):
    if not buffer:
        return "Unknown"
    return Counter(buffer).most_common(1)[0][0]

# ---------------- Webcam & FPS ----------------
cap = cv2.VideoCapture(0)

prev_time = time.time()
fps = 0

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- FPS Calculation --------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # -------- POSTURE --------
    posture, spine_angle = analyze_posture(frame)
    posture_buffer.append(posture)
    stable_posture = majority_vote(posture_buffer)

    # -------- EMOTION --------
    face = detect_face(frame)

    if face is not None:
        face_tensor = torch.tensor(preprocess_face(face)).float()
        emotion, conf = predict_emotion(face_tensor)
        emotion_buffer.append(emotion)
    else:
        emotion_buffer.clear()
        emotion = "Unknown"
        conf = 0.0

    stable_emotion = majority_vote(emotion_buffer)

    # Optional confidence thresholding
    if conf < 0.4:
        stable_emotion = "Uncertain"

    # -------- FEEDBACK --------
    feedback = generate_feedback(stable_posture, stable_emotion)

    # ---------------- UI OVERLAY ----------------
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (frame.shape[1] - 150, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Posture: {stable_posture}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Emotion: {stable_emotion} ({conf:.2f})",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Spine Angle: {int(spine_angle)}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Feedback: {feedback}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    cv2.imshow("AI Mirror", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
