import torch
import torch.nn.functional as F
from .model import EmotionCNN
import os

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

device = torch.device("cpu")
model = EmotionCNN().to(device)
MODEL_LOADED = False

def load_model(path):
    global MODEL_LOADED
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("[WARN] Emotion model weights not found. Using stub mode.")
        MODEL_LOADED = False
        return

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    MODEL_LOADED = True
    print("[OK] Emotion model loaded")

def predict_emotion(face_tensor):
    if not MODEL_LOADED:
        return "Neutral", 0.0

    with torch.no_grad():
        logits = model(face_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, dim=1)
        return EMOTIONS[idx.item()], confidence.item()
