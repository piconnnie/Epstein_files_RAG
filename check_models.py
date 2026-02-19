import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model_file = "models_list.txt"

if not api_key:
    with open(model_file, "w") as f:
        f.write("No API key found in .env")
else:
    genai.configure(api_key=api_key)
    with open(model_file, "w") as f:
        f.write(f"Checking models for API Key ending in ...{api_key[-4:]}\n")
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    f.write(f"- {m.name}\n")
        except Exception as e:
            f.write(f"Error listing models: {e}\n")
