import torch
from transformers import BitsAndBytesConfig, pipeline
import whisper
import gradio as gr
import os
from gtts import gTTS
import nltk
import re
import numpy as np

def setup_environment():
    try:
        # Download necessary NLTK data
        nltk.download('punkt')

        # Set up quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Set model ID and load the model
        global pipe
        model_id = "google/byt5-small"
        pipe = pipeline("text2text-generation",
                        model=model_id,
                        model_kwargs={"quantization_config": quantization_config})

        # Set CUDA path to environment variables
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
        os.environ["PATH"] += os.pathsep + cuda_path

        # Load Whisper model
        global model
        model = whisper.load_model("medium", device="cuda")

    except Exception as e:
        print(f"Error during setup: {e}")
        raise

# Function to transcribe audio
def transcribe(audio):
    if audio is None or audio == '':
        return ('', '', None)  # Return empty strings and None audio file

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    result_text = result.text

    return result_text

# Function to generate text description
def describe_text(input_text):
    prompt_instructions = f"Describe the context of the following text:\n{input_text}\n"
    prompt = prompt_instructions
    outputs = pipe(prompt, max_new_tokens=200)
    print("Outputs:", outputs)

    if outputs is not None and len(outputs[0]["generated_text"]) > 0:
        reply = outputs[0]["generated_text"]
    else:
        reply = "No response generated."
    return reply

# Function to handle audio input and generate description
def process_audio(audio_path):
    speech_to_text_output = transcribe(audio_path)
    description_output = describe_text(speech_to_text_output)
    return speech_to_text_output, description_output

if __name__ == "__main__":
    try:
        setup_environment()
        # Create the interface
        iface = gr.Interface(
            fn=process_audio,
            inputs=[
                gr.Audio(sources=["microphone", "upload"], type="filepath")
            ],
            outputs=[
                gr.Textbox(label="Speech to Text"),
                gr.Textbox(label="Description Output")
            ],
            title="Audio Transcription and Description",
            description="Upload an audio file to transcribe it into text and get a description of the context."
        )

        # Launch the interface
        iface.launch(debug=True)
    except Exception as e:
        print(f"Failed to start the application: {e}")