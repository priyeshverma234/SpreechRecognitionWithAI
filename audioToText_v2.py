import speech_recognition as sr
from pydub import AudioSegment
import os
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

class AudioToText:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.recognizer = sr.Recognizer()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained sentence transformer model

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
        return text

    def get_embeddings(self, text):
        return self.model.encode(text)

    def query_similarity(self, embeddings, query):
        query_embedding = self.model.encode(query)
        similarity_scores = util.pytorch_cos_sim(query_embedding, embeddings)
        return similarity_scores
    
    def answer_question(self, transcribed_text, question):
        # Generate embeddings for the transcribed text
        text_embeddings = self.get_embeddings(transcribed_text)
        
        # Generate embeddings for the question
        question_embedding = self.model.encode(question)
        
        # Calculate similarity scores
        similarity_scores = util.pytorch_cos_sim(question_embedding, text_embeddings)
        
        # Find the most relevant part of the text
        most_relevant_index = similarity_scores.argmax()
        sentences = transcribed_text.split('.')
        answer = sentences[most_relevant_index]
        
        return answer

# Example usage
if __name__ == "__main__":
    audio_file = "harvard.wav"  # Replace with your audio file path
    output_file = "transcription.txt"
    audio_to_text = AudioToText(audio_file)
    transcribed_text = audio_to_text.process_audio(output_file)
    print(f"Transcription written to {output_file}")

    # Generate embeddings for the transcribed text
    embeddings = audio_to_text.get_embeddings(transcribed_text)

    # Example query
    query = "What is the main topic of the audio?"
    answer = audio_to_text.answer_question(transcribed_text, query)
    print(f"Answer: {answer}")