from retrieval_utils import get_similar_chunks

import httpx

model = "gemma3:1b-it-qat"


def query_ollama(prompt: str, model: str) -> str:
    r = httpx.post(
        "http://ollama/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    return r.json()["response"]


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

    response = query_ollama(prompt, model)
    return response.text


if __name__ == "__main__":
    try:
        with open("query.txt", "r") as f:
            query = f.read().strip()
        print(answer_query(query, model=model))

    except IndexError as e:
        print(f"No query entered: {e}")
