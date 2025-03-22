import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
import torch
from transformers import pipeline

# 📌 Ses parametreleri
SAMPLE_RATE = 44100
SILENCE_DURATION = 1
CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 25000

# Sessizlik kontrolü
def is_silent(data, threshold=SILENCE_THRESHOLD):
    volume_norm = np.linalg.norm(data)
    return volume_norm < threshold

# Ses kaydı
def record_audio():
    st.info("🎙 Kayıt başladı. Sessizlik algılanınca duracak...")
    recording = []
    silent_duration = 0
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            data, _ = stream.read(chunk_samples)
            recording.append(data.copy())

            if is_silent(data):
                silent_duration += CHUNK_DURATION
            else:
                silent_duration = 0

            if silent_duration >= SILENCE_DURATION:
                break

    audio_data = np.concatenate(recording, axis=0)
    filename = "recorded_audio.wav"
    wav.write(filename, SAMPLE_RATE, audio_data)
    return filename

# Speech-to-text (Google)
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language="en")
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition sesi anlayamadı."
    except sr.RequestError as e:
        return f"Google Speech Recognition hizmetine erişilemiyor: {e}"

# Duygu analizi
def analyze_sentiment(text):
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        framework="pt"
    )
    return sentiment_pipeline(text)

# ---------------- STREAMLIT ----------------

st.set_page_config(page_title="🎤 İngilizce Sesli Duygu Analizi", layout="centered")
st.title("🎤 İngilizce Sesli Duygu Analizi")
st.markdown("Bu uygulama ile sesinizi kaydedebilir, metne dönüştürebilir ve otomatik duygu analizi yaptırabilirsiniz.")

if st.button("🎙 Kayıt Başlat"):
    with st.spinner("Kayıt yapılıyor..."):
        audio_file = record_audio()
    st.success("✅ Kayıt tamamlandı!")

    with st.spinner("Metne dönüştürülüyor..."):
        text = transcribe_audio(audio_file)
        st.markdown("**Transkripsiyon:**")
        st.code(text)

    with st.spinner("Duygu analizi yapılıyor..."):
        sentiment = analyze_sentiment(text)

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


