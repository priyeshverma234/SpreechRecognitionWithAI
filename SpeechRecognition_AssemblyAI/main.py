import requests
from api_02 import *

filename = 'd:/PythonLearning/SpeechRecognition/SpeechRecognition_AssemblyAI/Natural Language Processing Short.m4a'
audio_url = upload(filename)

save_transcript(audio_url, 'file_title')