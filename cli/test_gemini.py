import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
)
print(response.text)
if response.usage_metadata:
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
