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
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
import logging
from anyio import to_thread
from config import settings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


# ------------- timing functions ---------------#
def time_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(f"{func.__name__} failed after {elapsed:.3f}s")
            raise

    return wrapper


def time_async_generator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> AsyncGenerator:
        start = time.perf_counter()
        count = 0
        try:
            async for item in func(*args, **kwargs):
                count += 1
                yield item
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} yielded {count} items in {elapsed:.3f}s")
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                f"{func.__name__} failed after {elapsed:.3f}s (yielded {count} items)"
            )
            raise

    return wrapper


# ------------ Files extraction from source ---------------- #


@time_async
async def visit_url_and_decode_content(url):
    token = settings.github_api_token.get_secret_value()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, headers={"Authorization": f"Bearer {token}"}
            )

    except urllib.error.HTTPError as e:
        print(f"{e.status_code}: {e}")
        raise

    response.raise_for_status()
    data = response.json()

    if data.get("truncated"):
        raise RuntimeError(
            "Tree response was truncated — repo too large for single API call"
        )

    return await to_thread.run_sync(decode_content, data["content"])


def decode_content(encoded_str):
    return base64.b64decode(encoded_str).decode("utf-8")


@time_async_generator
async def visit_main_tree_and_extract_docs_files(repo_configs):
    token = settings.github_api_token.get_secret_value()

    # Get full file tree
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"https://api.github.com/repos/{repo_configs['owner']}/{repo_configs['repo']}/git/trees/{repo_configs['branch']}?recursive=1",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            time.sleep(0.01)
        except urllib.error.HTTPError as e:
            print(f"{e.status_code}: {e}")
            raise

    response.raise_for_status()
    data = response.json()

    if data.get("truncated"):
        raise RuntimeError(
            "Tree response was truncated — repo too large for single API call"
        )

    for item in data["tree"]:
        if (
            "docs" in item["path"]
            and item["path"].endswith(".md")
            and "api" not in item["path"]
        ):
            contents = await visit_url_and_decode_content(item["url"])

            yield Document(
                page_content=contents,
                metadata={
                    "source_path": item["path"],
                    "branch": repo_configs["branch"],
                },
            )


# Parent Child Splitter
CHUNK_SIZE = 2000  # parent size
CHILD_SIZE = 512

# --- Splitters ---
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n```", "\n\n", "\n"],
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_SIZE,
    chunk_overlap=50,
)


async def lazy_parent_splitter(file_document):

    for split in parent_splitter.split_documents([file_document]):
        split.metadata = {
            **file_document.metadata,
            **split.metadata,
            "parent_id": str(uuid.uuid4()),
        }
        yield split


async def lazy_child_splitter(parent):

    for split in child_splitter.split_documents([parent]):
        split.metadata = {
            **parent.metadata,
            **split.metadata,
        }
        yield split


async def lazy_load_batches(file_documents_generator, batch_size=256):
    parent_batch, child_batch = [], []

    async for document in file_documents_generator:
        parent_stream = lazy_parent_splitter(document)

        async for parent in parent_stream:
            parent_id = parent.metadata["parent_id"]

            parent_batch.append(
                {
                    "parent_id": parent_id,
                    "content": parent.page_content,
                    "metadata": parent.metadata,
                }
            )

            async for split in lazy_child_splitter(parent):
                split.metadata = {**parent.metadata, **split.metadata}
                child_batch.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": split.page_content,
                        "metadata": split.metadata,
                    }
                )

                if len(child_batch) == batch_size:
                    yield parent_batch, child_batch
                    parent_batch = []
                    child_batch = []

    if child_batch:
        yield parent_batch, child_batch


# ----- Writes all children and parents to json -------------#


# 1 GPU version
async def lazy_embed_chunks_to_json(
    repo_configs,
    model_name="msmarco-bert-base-dot-v5",
    children_path="children.json",
    parents_path="parents.json",
):
    document_stream = visit_main_tree_and_extract_docs_files(repo_configs)

    # --- Load single GPU ---
    logging.info("Loading embedding model on cuda:0...")
    model = SentenceTransformer(
        f"sentence-transformers/{model_name}",
        token=settings.hf_token.get_secret_value(),
    ).to("cuda:0")

    total_children = 0
    total_parents = 0
    all_parents = []

    with open(children_path, "w") as f:
        f.write("[\n")
        first = True

        async for parent_batch, child_batch in lazy_load_batches(
            document_stream, batch_size=256
        ):
            logging.info(f"Embedding batch (children={len(child_batch)}) → cuda:0")

            embeddings = await to_thread(
                lambda b=child_batch: model.encode(
                    [doc["content"] for doc in b],
                    normalize_embeddings=False,
                    show_progress_bar=False,
                ).tolist()
            )

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

        f.write("\n]")

    # --- write parents once at the end ---
    with open(parents_path, "w") as p:
        json.dump(all_parents, p, indent=2)

    logging.info(f"Saved {total_children} children → {children_path}")
    logging.info(f"Saved {total_parents} parents  → {parents_path}")
    return total_children, total_parents


@time_async
async def delete_all_embeddings(database_url, collection_name="developer_docs"):
    """Truncate all embeddings from the PGVector collection without dropping it."""

    engine = create_async_engine(database_url)
    async with engine.begin() as conn:
        # Delete all rows in the embedding table belonging to this collection
        await conn.execute(
            text("""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection
                WHERE name = :collection_name
            )
        """),
            {"collection_name": collection_name},
        )

    await engine.dispose()

    print(f"Cleared all embeddings from collection '{collection_name}'.")
