import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from transformers import pipeline
import tempfile

# Sayfa başlığı ve açıklama
st.set_page_config(page_title="🎤 Voice Sentiment Analysis")
st.title("🎤 Voice Sentiment Analysis")
st.markdown("Welcome! Click the microphone icon below to record your speech and wait for the analysis to complete.")

# Ses kaydı başlat
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=44100)

if audio_bytes:
    st.success("✅ Audio received!")

    # Geçici .wav dosyasına yaz
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Speech-to-text (Google API)
    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        st.markdown("### 📄 Transcription")
        st.code(text)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        text = ""

    # Duygu Analizi
    if text:
        sentiment = pipeline("sentiment-analysis")(text)[0]

        label = sentiment["label"]
        score = sentiment["score"] * 100

        emoji = "😐"
        color = "#f0f0f0"
        if label.lower() == "positive":
            emoji = "😊"
            color = "#d4edda"
        elif label.lower() == "negative":
            emoji = "😠"
            color = "#f8d7da"

        st.markdown("### 💬 Sentiment Analysis")
        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px;">
                <strong>Sentiment:</strong> {label} {emoji}<br>
                <strong>Confidence:</strong> {score:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )











