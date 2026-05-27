import os
import functools
import time
import json
import uuid
import urllib
from typing import AsyncGenerator
import base64
import httpx
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from anyio import to_thread
import asyncio
from itertools import islice
import anyio
import torch 
from typing import List

from extraction_utils import stream_documents_in_batches, lazy_load_batches, lazy_parent_splitter, lazy_child_splitter

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



# ----- Writes all children and parents to json -------------#


async def _next_batch(gen):
    """Pull next batch from async generator, return None when exhausted."""
    try:
        return await gen.__anext__()
    except StopAsyncIteration:
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHILD_PATH = os.path.join(BASE_DIR, "..", "data", "children.json")
PARENT_PATH =  os.path.join(BASE_DIR, "..", "data", "parent.json")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

async def lazy_embed_chunks_to_json(
    repo_configs,
    model_name="msmarco-bert-base-dot-v5",
    children_path="children.json",
    parents_path="parents.json",
    fetch_batch_size=10,
):
    documents_stream = stream_documents_in_batches(repo_configs, fetch_batch_size)

    # --- Load single GPU ---
    logging.info("Loading embedding model on cuda:0...")
    token = SecretStr(os.environ.get("HF_TOKEN", None))

    embedder = MSMarcoEmbeddings(model_name, token=token.get_secret_value())

    model = embedder.model.to(device)


    # Get first batch of chunks
    batch_gen = lazy_load_batches(documents_stream)
    current = await _next_batch(batch_gen)

    if current is None:
        logging.info("No documents found.")
        raise RuntimeError("No documents found to embed.")

    # Start embedding current batch
    total_children, total_parents = 0, 0
    all_parents = []

        # Get username
    print(os.getenv("USER") or os.getenv("USERNAME"))

    # Get numeric user ID
    print(os.getuid())   # Unix only

    # Get group ID
    print(os.getgid())   # Unix only
    print(CHILD_PATH)
    with open(CHILD_PATH, "w") as f:
        f.write("[\n")
        first = True

        while current is not None:
            parent_batch, child_batch = current
            print(f"Embedding batch (children={len(child_batch)}) → {device}")

            # Before embedding starts, create task to fetch next batch of documents
            # while embeddings runs in thread, this task gets next batch concurrently
            next_batch_task = asyncio.create_task(_next_batch(batch_gen))

            embeddings = await to_thread.run_sync(
                lambda b=child_batch: model.encode(
                    [doc["content"] for doc in b],
                    normalize_embeddings=False,
                    show_progress_bar=False,
                ).tolist()
            )

            # Write embedded results to disk
            all_parents.extend(parent_batch)
            for doc, emb in zip(child_batch, embeddings):
                if not first:
                    f.write(",\n")
                f.write(json.dumps({**doc, "embedding": emb}))
                first = False

            f.flush()
            os.fsync(f.fileno())

            total_children += len(child_batch)
            total_parents += len(parent_batch)
            logging.info(f"Flushed to disk. Total children so far: {total_children}")

            # Next batch is likely already ready (or nearly so)
            current = await next_batch_task

        f.write("\n]")

    # --- write parents once at the end ---
    with open(PARENT_PATH, "w") as p:
        json.dump(all_parents, p, indent=2)

    logging.info(f"Saved {total_children} children → {children_path}")
    logging.info(f"Saved {total_parents} parents  → {parents_path}")

    return total_children, total_parents



