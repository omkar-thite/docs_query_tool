import os
import sys
from anyio import to_thread
from typing import List
from sentence_transformers import SentenceTransformer


embed_model_name = "msmarco-bert-base-dot-v5"


class MSMarcoEmbeddings:
    def __init__(self, embed_model_name, token=None):
        safe_embed_model_name = embed_model_name.replace("/", "_")
        cache_dir = "./local_model_cache"
        self.model_path = os.path.join(cache_dir, safe_embed_model_name)

        if os.path.exists(self.model_path):
            print(f"Loading model from local directory: {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
        else:
            print(
                f"Loading model from HuggingFace: sentence-transformers/{embed_model_name}"
            )
            self.model = SentenceTransformer(
                f"sentence-transformers/{embed_model_name}", token=token
            )

            print(f"Saving model to local directory: {self.model_path}")
            os.makedirs(cache_dir, exist_ok=True)
            self.model.save(self.model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=False,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text,
            normalize_embeddings=False,
            convert_to_numpy=True,
        ).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return await to_thread.run_sync(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return await to_thread.run_sync(self.embed_query, text)


embedder = MSMarcoEmbeddings(embed_model_name)


def get_similar_chunks(query: str, k: int = 3) -> dict:

    if not query:
        print("No query provided")
        return

    # with get_conn(DB_CONFIG) as conn:
    #     query_embedding = embedder.embed_query(query)

    #     db = http

    #     rows = conn.execute(
    #         """
    #         SELECT
    #             c.id,
    #             c.parent_id,
    #             c.content AS child_content,
    #             p.content AS parent_content,
    #             p.source_path,
    #             c.embedding <#> %s::vector  AS distance
    #         FROM document_chunks  c
    #         JOIN document_parents p
    #         ON c.parent_id = p.id
    #         ORDER BY distance
    #         LIMIT 5;
    #     """,
    #         (query_embedding,),
    #     ).fetchall()

    #     responses = []
    #     for row in rows[:k]:
    #         responses.append(row[2])

    #     return {"query": query, "responses": responses}


def main():
    query = sys.argv[1]
    get_similar_chunks(query)


if __name__ == "__main__":
    main()
