import os
import httpx
import urllib
import base64
import uuid
import asyncio
from itertools import islice
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from anyio import to_thread


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

# ------------ Files extraction from source ---------------- #

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



async def visit_url_and_decode_content(url):
    token = os.environ.get("GITHUB_API_TOKEN", None)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, headers=headers, timeout=10
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


async def extract_main_tree(repo_configs):
    token = os.environ.get("GITHUB_API_TOKEN", None)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Get full file tree
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"https://api.github.com/repos/{repo_configs['owner']}/{repo_configs['repo']}/git/trees/{repo_configs['branch']}?recursive=1",
                headers=headers,
                timeout=10,
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

    return [
        item
        for item in data["tree"]
        if "docs" in item["path"]
        and item["path"].endswith(".md")
        and "api" not in item["path"]
    ]


async def stream_documents_in_batches(repo_configs: dict, batch_size=10):

    tree = await extract_main_tree(repo_configs)
    it = iter(tree)

    while batch := list(islice(it, batch_size)):
        contents_list = await asyncio.gather(
            *[visit_url_and_decode_content(item["url"]) for item in batch]
        )

        for item, contents in zip(batch, contents_list):
            yield Document(
                page_content=contents,
                metadata={
                    "source_path": item["path"],
                    "branch": repo_configs["branch"],
                },
            )


