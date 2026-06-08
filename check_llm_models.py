# check_llm_models.py
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for model in client.models.list():
    if "generateContent" in [m for m in (model.supported_actions or [])]:
        print(model.name)