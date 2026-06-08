# check_dims.py
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

result = client.models.embed_content(
    model="gemini-embedding-2",
    contents="テスト",
)
print(f"gemini-embedding-2: {len(result.embeddings[0].values)}次元")