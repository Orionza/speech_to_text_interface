import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import wave
import tempfile
import speech_recognition as sr
from transformers import pipeline

# 🛠 PORT tanımı: Render için gerekli!
os.environ["PORT"] = os.environ.get("PORT", "8501")

# Streamlit sayfa ayarları
st.set_page_config(page_title="🎤 Voice Sentiment Analysis", layout="centered")
st.title("🎤 Voice Sentiment Analysis")
st.markdown("Welcome! Click **Start** and speak into your microphone. Then stop and wait for the analysis.")

st.warning("⚠️ This works best in Chrome. Please allow microphone access when asked.")

# Mikrofon verisi için sınıf
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

# Streamlit WebRTC bileşeni
ctx = webrtc_streamer(
    key="speech-demo",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Kaydı durdurunca işle
if ctx and not ctx.state.playing and ctx.audio_processor:
    st.success("✅ Recording finished. Processing...")

    # WAV dosyasına yaz
    audio_data = np.concatenate(ctx.audio_processor.frames, axis=0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        with wave.open(tmp_wav.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(48000)
            wf.writeframes(audio_data.tobytes())
        audio_file_path = tmp_wav.name

    # 🎙 Transkripsiyon
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        st.markdown("**Transcription:**")
        st.code(text)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        text = ""

    # 😊 Duygu Analizi
    if text:
        st.markdown("**Sentiment Analysis Result:**")
        sentiment = pipeline("sentiment-analysis")(text)[0]
        label = sentiment["label"]
        score = sentiment["score"] * 100

        color = "#d1ecf1"
        emoji = "😐"
        if label.lower() == "positive":
            color = "#d4edda"
            emoji = "😊"
        elif label.lower() == "negative":
            color = "#f8d7da"
            emoji = "😠"

        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; border: 1px solid #ccc;">
                <p><strong>Sentiment:</strong> {label.title()} {emoji}</p>
                <p><strong>Confidence:</strong> {score:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )








