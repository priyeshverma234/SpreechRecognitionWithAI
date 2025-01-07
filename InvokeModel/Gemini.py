import google.generativeai as genai
import os
from dotenv import load_dotenv
import pathlib
import textwrap
from IPython.display import Markdown

class GeminiModel:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)

    def to_markdown(self, text):
        text = text.replace('•', '  *')
        return textwrap.indent(text, '> ', predicate=lambda _: True)

    def trigger_model(self, prompt):
        # for m in genai.list_models():
        #     if 'generateContent' in m.supported_generation_methods:
        #         print(m.name)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        text = response.text 
        print("Gemini Response: ", self.to_markdown(text))  # Use print instead of display
        return text

# Example usage:
if __name__ == "__main__":
    gemini = GeminiModel()
    gemini.trigger_model("What is the meaning of life?")