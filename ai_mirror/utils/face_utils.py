import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    return gray[y:y+h, x:x+w]

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 1, 48, 48)
    return face
