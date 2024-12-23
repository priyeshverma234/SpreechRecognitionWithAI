import streamlit as st
import speech_recognition as sr
import tempfile
import os

# Initialize recognizer
recognizer = sr.Recognizer()

# Streamlit UI
st.title("Speech to Text Application")

if 'recording' not in st.session_state:
    st.session_state.recording = False

def start_recording():
    st.session_state.recording = True

def stop_recording():
    st.session_state.recording = False

st.button("Start Recording", on_click=start_recording)
st.button("Stop Recording", on_click=stop_recording)

if st.session_state.recording:
    st.write("Recording... Please speak into the microphone.")
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=5)
        st.session_state.audio_data = audio_data

if not st.session_state.recording and 'audio_data' in st.session_state:
    st.write("Processing audio...")
    try:
        text = recognizer.recognize_google(st.session_state.audio_data)
        st.write("You said:", text)
    except sr.UnknownValueError:
        st.write("Sorry, could not understand audio.")
    except sr.RequestError as e:
        st.write(f"Error: Could not request results from Google Speech Recognition service; {e}")