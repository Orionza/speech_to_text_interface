import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from transformers import pipeline
import tempfile

st.set_page_config(page_title="🎤 Voice Sentiment Analysis")
st.title("🎤 Voice Sentiment Analysis")
st.markdown("Click the microphone below to record your voice. Wait and let us analyze it!")

# Mikrofonla kayıt
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=44100)

if audio_bytes:
    st.success("✅ Audio recorded!")

    # Geçici WAV dosyası
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Transkripsiyon
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

    # Duygu analizi
    if text:
        sentiment_pipeline = pipeline("sentiment-analysis")
        result = sentiment_pipeline(text)[0]

        label = result["label"]
        score = result["score"] * 100

        emoji = "😐"
        if label.lower() == "positive":
            emoji = "😊"
        elif label.lower() == "negative":
            emoji = "😠"

        st.markdown("### 💬 Sentiment Analysis")
        st.markdown(f"**Sentiment:** {label} {emoji}  \n**Confidence:** {score:.2f}%")










