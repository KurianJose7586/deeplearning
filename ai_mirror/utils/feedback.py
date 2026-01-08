def generate_feedback(posture, emotion):
    """
    Rule-based multimodal feedback system
    """

    # Normalize inputs
    posture = posture.lower()
    emotion = emotion.lower()

    # --- High-priority posture issues ---
    if posture == "slouched":
        if emotion in ["sad", "fear", "angry"]:
            return "You look tired. Straighten your posture 😊"
        elif emotion == "neutral":
            return "Try sitting upright for better comfort 🙂"
        else:
            return "Good mood! Just fix your posture 👍"

    # --- Leaning cases ---
    if posture == "leaning":
        return "Try to sit straight to avoid strain."

    # --- Good posture cases ---
    if posture == "good":
        if emotion in ["happy", "surprise"]:
            return "You look confident and energetic 💪"
        elif emotion == "neutral":
            return "Good posture! Keep it up 👌"
        else:
            return "Nice posture. Take a deep breath 😌"

    # --- Fallback ---
    return "Stay relaxed and comfortable."
