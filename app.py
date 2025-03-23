import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from transformers import pipeline
import tempfile

# Sayfa başlığı ve tema
st.set_page_config(page_title="🎤 Voice Sentiment Analysis", layout="centered")
st.title("🎤 Voice Sentiment Analysis")
st.markdown("Welcome! Click the microphone icon below to record your speech and wait for the analysis to complete.")

# Mikrofonla ses kaydı
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=44100)

if audio_bytes:
    st.success("✅ Audio received!")

    # Geçici WAV dosyası
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Transkripsiyon (Google Speech Recognition)
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

    # Duygu Analizi (CardiffNLP Modeli)
    if text:
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
        result = sentiment_pipeline(text)[0]

        label = result["label"].lower()
        score = result["score"] * 100

        # Renk ve emoji (koyu temaya uygun)
        if label == "positive":
            bg_color = "#0d4625"  # koyu yeşil
            emoji = "😊"
        elif label == "negative":
            bg_color = "#5c1b1b"  # koyu kırmızı
            emoji = "😠"
        else:
            bg_color = "#1b3d5c"  # koyu mavi
            label = "neutral"
            emoji = "😐"

        # Duygu analizi sonucu gösterimi
        st.markdown("### 💬 Sentiment Analysis")
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; color: white; padding: 20px; border-radius: 10px;">
                <p style="font-size: 18px;"><strong>Sentiment:</strong> {label.upper()} {emoji}</p>
                <p><strong>Confidence:</strong> {score:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )











