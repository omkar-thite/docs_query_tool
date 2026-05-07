import sys

from google import genai
from google.genai import types


from retrieval import get_similar_chunks
from config import llm_api_settings

API_KEY = llm_api_settings.gemini_api_key.get_secret_value()
client = genai.Client(api_key=API_KEY)
model = "gemini-3-flash-preview"


def get_completion(prompt, model="gemma-3-1b-it"):
    contents = [{"role": "user", "parts": [{"text": prompt}]}]

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(temperature=0),
        contents=contents,
    )

    return response.text


def build_rag_prompt(query: str, chunks: list[str]) -> str:
    context = "\n---\n".join(chunks)  # separator between chunks
    return (
        f"### Context\n{context}\n\n"
        f"### Question\n{query}\n\n"
        f"Answer using only the context above. If unknown, say 'Not found'.\n\n"
        f"If there are code snippets in the context, include them in your answer and format them as markdown.\n\n"
        f"### Answer\n"  # model continues from here
    )


def answer_query(
    query: str,
    model: str,
    k: int = 3,
):
    chunks = get_similar_chunks(query)["responses"]
    prompt = build_rag_prompt(query, chunks)

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(temperature=0),
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )
    return response.text


if __name__ == "__main__":
    # # List all models available for your account
    # for model in client.models.list():
    #     print(f"Model: {model.name}")

    query = sys.argv[1]
    print(answer_query(query, model=model))
