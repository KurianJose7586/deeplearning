import streamlit as st
import subprocess
import sys

# ⚠️ MUST BE FIRST STREAMLIT CALL
st.set_page_config(page_title="AI Mirror", layout="centered")

st.title("🪞 AI Mirror")
st.write("Real-time posture & emotion feedback system")

if st.button("Start AI Mirror"):
    st.write("Launching camera...")
    subprocess.Popen([sys.executable, "-m", "realtime.mirror"])

st.markdown("---")
st.write("📊 Features:")
st.write("- Real-time posture analysis")
st.write("- Emotion recognition with confidence")
st.write("- Temporal smoothing")
st.write("- Human-centric feedback")
