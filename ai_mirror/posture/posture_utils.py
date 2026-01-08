import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------- MediaPipe Pose Setup --------

MODEL_PATH = "pose_landmarker_lite.task"

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = PoseLandmarker.create_from_options(options)

# -------- Utility Functions --------

def angle_between(a, b, c):
    """Angle at point b between points a and c"""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba * mag_bc == 0:
        return 0

    cosine = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


# -------- Main Posture Analysis --------

def analyze_posture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    posture = "Unknown"
    spine_angle = 0

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        h, w, _ = frame.shape

        def to_xy(lm):
            return (int(lm.x * w), int(lm.y * h))

        nose = to_xy(landmarks[0])
        l_sh = to_xy(landmarks[11])
        r_sh = to_xy(landmarks[12])
        l_hip = to_xy(landmarks[23])
        r_hip = to_xy(landmarks[24])

        mid_shoulder = (
            (l_sh[0] + r_sh[0]) // 2,
            (l_sh[1] + r_sh[1]) // 2
        )
        mid_hip = (
            (l_hip[0] + r_hip[0]) // 2,
            (l_hip[1] + r_hip[1]) // 2
        )

        spine_angle = angle_between(nose, mid_shoulder, mid_hip)

        if spine_angle > 160:
            posture = "Good"
            color = (0, 255, 0)
        elif spine_angle > 140:
            posture = "Leaning"
            color = (0, 255, 255)
        else:
            posture = "Slouched"
            color = (0, 0, 255)

        # draw landmarks ONLY
        for lm in landmarks:
            x, y = to_xy(lm)
            cv2.circle(frame, (x, y), 3, color, -1)

    return posture, spine_angle
