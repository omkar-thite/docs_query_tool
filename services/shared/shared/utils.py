class MSMarcoEmbeddings:
    def __init__(self, embed_model_name: str, token=None):
        safe_embed_model_name = embed_model_name.replace("/", "_")
        cache_dir = "data/local_model_cache"
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


class SecretStr:
    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self):
        return "SecretStr('**********')"

    def __str__(self):
        return "**********"

    def __bool__(self):
        return bool(self._value)

