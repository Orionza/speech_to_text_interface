import streamlit as st
from audio_recorder_streamlit import audio_recorder
import scipy.io.wavfile as wav
import speech_recognition as sr
from transformers import pipeline
import torch

# Streamlit başlık
st.set_page_config(page_title="🎤 Sesli Duygu Analizi")
st.title("🎤 İngilizce Sesli Duygu Analizi")
st.markdown("Tarayıcı mikrofonunuzu kullanarak ses kaydedin, konuşmanız analiz edilsin!")

# Ses kaydı başlat
audio_bytes = audio_recorder(text=" Kayıt için mikrofona tıklayınız", icon_size="2x")


if audio_bytes:
    filename = "kayit.wav"
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    st.success("✅ Kayıt alındı!")

    # Transkripsiyon
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language="en")
        st.markdown("**Transkripsiyon:**")
        st.code(text)
    except Exception as e:
        st.error(f"Transkripsiyon hatası: {e}")
        text = ""

    # Duygu analizi
    if text:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            framework="pt"
        )

        sentiment = sentiment_pipeline(text)
        label = sentiment[0]['label'].lower()
        score = sentiment[0]['score']
        score_percent = f"%{score * 100:.2f}"

        if label == "positive":
            color = "#d4edda"
            emoji = "😊"
            label_text = "Pozitif"
        elif label == "negative":
            color = "#f8d7da"
            emoji = "😠"
            label_text = "Negatif"
        else:
            color = "#d1ecf1"
            emoji = "😐"
            label_text = "Nötr"

        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; border: 1px solid #ccc;">
                <h4 style="margin-bottom: 10px;">🔍 Duygu Analizi Sonucu</h4>
                <p style="font-size: 18px;"><strong>Duygu:</strong> {label_text} {emoji}</p>
                <p style="font-size: 18px;"><strong>Güven Skoru:</strong> {score_percent}</p>
            </div>
            """,
            unsafe_allow_html=True
        )






