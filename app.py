import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import wave
import speech_recognition as sr
from transformers import pipeline
import tempfile

st.set_page_config(page_title="🎤 Voice Sentiment Analysis", layout="centered")
st.title("🎤 Voice Sentiment Analysis")
st.markdown("Welcome! Click 'Start' and speak into your microphone. Then stop recording and wait for analysis.")

st.warning("⚠️ This demo works best in Chrome. Allow microphone access when prompted.")

# AudioProcessor to collect voice frames
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        data = frame.to_ndarray()
        self.frames.append(data)
        return frame

# Stream audio
ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# After recording
if ctx and ctx.state.playing:
    st.info("🎙 Recording... Speak now!")
elif ctx and not ctx.state.playing and ctx.audio_processor:
    st.success("✅ Recording finished. Processing...")

    # Save audio to WAV
    audio_data = np.concatenate(ctx.audio_processor.frames, axis=0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        with wave.open(tmp_wav.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio_data.tobytes())
        audio_file = tmp_wav.name

    # Transcribe
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        st.markdown("**Transcription:**")
        st.code(text)
    except Exception as e:
        st.error(f"Transcription error: {e}")
        text = ""

    # Sentiment analysis
    if text:
        sentiment_pipeline = pipeline("sentiment-analysis")
        result = sentiment_pipeline(text)[0]

        label = result["label"]
        score = result["score"] * 100

        color = {
            "POSITIVE": "#d4edda",
            "NEGATIVE": "#f8d7da",
            "NEUTRAL": "#d1ecf1"
        }.get(label.upper(), "#eeeeee")

        emoji = {
            "POSITIVE": "😊",
            "NEGATIVE": "😠",
            "NEUTRAL": "😐"
        }.get(label.upper(), "🤔")

        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; border: 1px solid #ccc;">
                <h4>🔍 Sentiment Analysis Result</h4>
                <p><strong>Sentiment:</strong> {label.title()} {emoji}</p>
                <p><strong>Confidence:</strong> {score:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )







