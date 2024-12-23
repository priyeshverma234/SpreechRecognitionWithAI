import speech_recognition as sr
from pydub import AudioSegment
import os

class AudioToText:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.recognizer = sr.Recognizer()

    def convert_audio_to_wav(self):
        audio_format = self.audio_file.split('.')[-1]
        audio = AudioSegment.from_file(self.audio_file, format=audio_format)
        wav_file = "temp.wav"
        audio.export(wav_file, format="wav")
        return wav_file

    def transcribe_audio(self, wav_file):
        with sr.AudioFile(wav_file) as source:
            audio_data = self.recognizer.record(source)
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, could not understand audio."
        except sr.RequestError as e:
            return f"Error: Could not request results from Google Speech Recognition service; {e}"

    def write_text_to_file(self, text, output_file):
        with open(output_file, 'w') as file:
            file.write(text)

    def process_audio(self, output_file):
        wav_file = self.convert_audio_to_wav()
        text = self.transcribe_audio(wav_file)
        self.write_text_to_file(text, output_file)
        os.remove(wav_file)  # Clean up temporary WAV file

# Example usage
if __name__ == "__main__":
    audio_file = "harvard.wav"  # Replace with your audio file path
    output_file = "transcription.txt"
    audio_to_text = AudioToText(audio_file)
    audio_to_text.process_audio(output_file)
    print(f"Transcription written to {output_file}")