import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

def rag(query: str, docs: list[str]) -> str:
    if not docs:
        return "No documents provided"

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text
    except ClientError as e:
        if e.status_code == 429:
            return f"Error: API rate limit exceeded. Please try again later. ({e.status_code} RESOURCE_EXHAUSTED)"
        raise