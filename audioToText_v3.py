import speech_recognition as sr
from pydub import AudioSegment
import os
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

class AudioToText:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.recognizer = sr.Recognizer()
        self.hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")

        # Set the Hugging Face token as an environment variable
        os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hugging_face_token

        # Initialize HuggingFacePipeline
        self.hf_pipeline = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 150},
            device=-1
        )

        # Initialize PromptTemplate
        template = """Context: {context}

        Prompt: {prompt}

        Answer: Let's think step by step."""
        self.prompt = PromptTemplate.from_template(template)

        # Create the chain
        self.chain = self.prompt | self.hf_pipeline

    def convert_audio_to_wav(self):
        audio_format = self.audio_file.split('.')[-1]
        audio = AudioSegment.from_file(self.audio_file, format=audio_format)
        wav_file = "temp.wav"
        audio.export(wav_file, format="wav")
        return wav_file

    def split_audio(self, wav_file, chunk_length_ms=60000):  # 1 minutes
        audio = AudioSegment.from_wav(wav_file)
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        return chunks

    def transcribe_audio_chunk(self, audio_chunk):
        with sr.AudioFile(audio_chunk) as source:
            audio_data = self.recognizer.record(source)
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, could not understand audio."
        except sr.RequestError as e:
            return f"Error: Could not request results from Google Speech Recognition service; {e}"

    def transcribe_audio(self, wav_file):
        chunks = self.split_audio(wav_file)
        full_text = ""
        for i, chunk in enumerate(chunks):
            chunk_file = f"chunk{i}.wav"
            chunk.export(chunk_file, format="wav")
            text = self.transcribe_audio_chunk(chunk_file)
            full_text += text + " "
            os.remove(chunk_file)  # Clean up chunk file
        return full_text

    def write_text_to_file(self, text, output_file):
        with open(output_file, 'w') as file:
            file.write(text)

    def process_audio(self, output_file):
        wav_file = self.convert_audio_to_wav()
        text = self.transcribe_audio(wav_file)
        self.write_text_to_file(text, output_file)
        os.remove(wav_file)  # Clean up temporary WAV file
        return text

    def generate_output_with_context(self, text, prompt):
        # Create the input for the model
        input_data = {"context": text, "prompt": prompt}
        
        # Generate output using the chain
        response = self.chain.invoke(input_data)
        return response

# Example usage
if __name__ == "__main__":
    audio_file = "SampleAudioFiles/Committee of the Whole 2024-12-10.mp3"  # Replace with your audio file path
    output_file = "SampleOutput/transcription.txt"
    audio_to_text = AudioToText(audio_file)
    transcribed_text = audio_to_text.process_audio(output_file)
    print(f"Transcription written to {output_file}")

    # Example prompt
    prompt = "What is the main topic of the audio?"
    generated_output = audio_to_text.generate_output_with_context(transcribed_text, prompt)
    print(f"Generated Output: {generated_output}")