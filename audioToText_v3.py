import speech_recognition as sr
from pydub import AudioSegment
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

class AudioToText:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.recognizer = sr.Recognizer()
        self.bert_model = AutoModel.from_pretrained("bert-base-cased")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.quantized_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def generate_output_with_context(self, embeddings, prompt):
        context = " ".join(map(str, embeddings))
        input_text = f"context: {context} prompt: {prompt}"
        inputs = self.tokenizer(input_text, return_tensors='pt').to("cuda")

        # Generate output
        outputs = self.quantized_model.generate(**inputs, max_new_tokens=150)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Example usage
if __name__ == "__main__":
    audio_file = "harvard.wav"  # Replace with your audio file path
    output_file = "transcription.txt"
    audio_to_text = AudioToText(audio_file)
    transcribed_text = audio_to_text.process_audio(output_file)
    print(f"Transcription written to {output_file}")

    # Generate embeddings for the transcribed text
    embeddings = audio_to_text.get_embeddings(transcribed_text)

    # Example prompt
    prompt = "What is the main topic of the audio?"
    generated_output = audio_to_text.generate_output_with_context(embeddings, prompt)
    print(f"Generated Output: {generated_output}")