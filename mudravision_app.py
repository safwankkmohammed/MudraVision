import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS
import time
import os

# -------------------------------
# Load Mudra Model
# -------------------------------
gesture_model = tf.keras.models.load_model("mudra_model1.keras")

MUDRA_CLASSES = ["Pataka", "Tripataka", "Ardhapataka", "Kartarimukha"]

# Mudra meanings (Viniyoga)
mudra_viniyoga = {
    "Pataka": "Pataka mudra is used for flag, clouds, forest, and denial.",
    "Tripataka": "Tripataka mudra is used for crown, tree, lamp, and arrows.",
    "Ardhapataka": "Ardhapataka mudra is used for leaves, knife, and separation.",
    "Kartarimukha": "Kartarimukha mudra is used for scissors, opposition, and lightning."
}

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="MudraVision", layout="wide")

st.title("🪷 MudraVision – Bharatanatyam Mudra Recognition")
st.markdown("Real-time AI Mudra Detection + Meaning + Audio Output")

frame_window = st.image([])

st.sidebar.title("🎛 Controls")

# -------------------------------
# Session State Setup
# -------------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "last_mudra" not in st.session_state:
    st.session_state.last_mudra = ""

# -------------------------------
# Buttons
# -------------------------------
start_btn = st.sidebar.button("▶ Start Mudra Recognition")
stop_btn = st.sidebar.button("⛔ Stop System")

if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False
    st.session_state.last_mudra = ""

# -------------------------------
# Confidence Threshold
# -------------------------------
confidence_threshold = 0.75

# -------------------------------
# Webcam Start
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# MAIN LOOP
# -------------------------------
if st.session_state.running:

    st.sidebar.success("System Running ✅")

    while st.session_state.running:

        ret, frame = cap.read()
        if not ret:
            st.error("❌ Webcam not accessible")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------------
        # Gesture Prediction
        # -------------------------------
        resized = cv2.resize(rgb_frame, (224, 224))
        img_array = np.expand_dims(resized / 255.0, axis=0)

        prediction = gesture_model.predict(img_array, verbose=0)

        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]

        # -------------------------------
        # Apply Confidence Threshold
        # -------------------------------
        if confidence < confidence_threshold:
            mudra_name = "No Gesture"
        else:
            mudra_name = MUDRA_CLASSES[class_id]

        # -------------------------------
        # Display in Sidebar
        # -------------------------------
        st.sidebar.write(f"### 🖐 Detected Mudra: {mudra_name}")
        st.sidebar.write(f"Confidence: **{confidence*100:.2f}%**")

        # -------------------------------
        # Speak Meaning Only When Changed
        # -------------------------------
        if mudra_name != "No Gesture":

            if mudra_name != st.session_state.last_mudra:

                st.session_state.last_mudra = mudra_name

                meaning = mudra_viniyoga[mudra_name]

                # Unique audio filename (prevents duplicate error)
                audio_file = f"mudra_{int(time.time())}.mp3"

                # Generate Speech
                tts = gTTS(text=meaning, lang="en")
                tts.save(audio_file)

                # Play Audio
                st.sidebar.audio(audio_file)

                st.sidebar.info(f"📖 Meaning: {meaning}")

        # -------------------------------
        # Display Mudra Name on Webcam
        # -------------------------------
        cv2.putText(
            frame,
            f"{mudra_name} ({confidence*100:.1f}%)",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show Webcam Frame
        frame_window.image(frame, channels="BGR")

    cap.release()

else:
    st.sidebar.warning("System Stopped ⛔")
    st.info("Click ▶ Start Mudra Recognition to begin.")
