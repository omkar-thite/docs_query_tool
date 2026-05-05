import functools
import time
from typing import List, AsyncGenerator
import base64
import httpx
from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from config import settings
from sentence_transformers import SentenceTransformer
import logging
from langchain_core.embeddings import Embeddings
from anyio import to_thread
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
logger = logging.getLogger(__name__)


headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
           ]


chunk_size = 50
chunk_overlap = 15

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers = False)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " "], add_start_index=True
)



class MSMarcoEmbeddings(Embeddings):

    def __init__(self, model_name, token=None):
        print(model_name)
        self.model = SentenceTransformer(f'sentence-transformers/{model_name}', token=token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=False).tolist()


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
            logger.exception(f"{func.__name__} failed after {elapsed:.3f}s (yielded {count} items)")
            raise
    return wrapper



@time_async
async def visit_url_and_decode_content(url):   
    token = settings.github_api_token.get_secret_value()

    async with httpx.AsyncClient() as client:
        response = await client.get(url, 
                           headers={"Authorization": f"Bearer {token}"}
                        )
        response.raise_for_status()
        data = response.json()
    
    if data.get("truncated"):
        raise RuntimeError("Tree response was truncated — repo too large for single API call")
            
    return await to_thread.run_sync(decode_content, data['content'])


def decode_content(encoded_str):
    return base64.b64decode(encoded_str).decode('utf-8')


@time_async_generator    
async def visit_main_tree_and_extract_md_files(repo_configs):
    token = settings.github_api_token.get_secret_value()

    # Get full file tree
    async with httpx.AsyncClient() as client:
        response = await client.get( f"https://api.github.com/repos/{repo_configs['owner']}/{repo_configs['repo']}/git/trees/{repo_configs['branch']}?recursive=1",
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=10
                            )
    
    response.raise_for_status()
    data = response.json()
    
    if data.get("truncated"):
        raise RuntimeError("Tree response was truncated — repo too large for single API call")
        
    for item in data['tree']:
        if item['path'].endswith('.md'):
            contents = await visit_url_and_decode_content(item['url'])

            yield Document(
                page_content=contents, 
                metadata={
                    "source_path": item['path'],
                    "repo": repo_configs['repo'],
                    "owner": repo_configs['owner'],
                    "branch": repo_configs['branch'],
                    "github_url": f"https://github.com/{repo_configs['owner']}/{repo_configs['repo']}/blob/{repo_configs['branch']}/{item['path']}"
                }
            )



@time_async_generator 
async def lazy_markdown_splitter(documents_generator):    
    async for doc in documents_generator:
        for split in markdown_splitter.split_text(doc.page_content):
            split.metadata = {**doc.metadata, **split.metadata}
            yield split


@time_async_generator 
async def lazy_split_text_in_chunk(md_splits_generator):
    '''split: Mardown Split Document'''
    async for split in md_splits_generator:
        # Note that split_document expects an iterator containing Documents, so pass split in a list
        for chunk in text_splitter.split_documents([split]):
            yield chunk


@time_async_generator 
async def lazy_load_batches(async_iterable, batch_size):
    batch = []
    async for item in async_iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
        if batch:
            yield batch 


@time_async
async def lazy_insert_chunk_batches(repo_configs, emb_model, database_url, batch_size=32):
    
    document_stream = visit_main_tree_and_extract_md_files(repo_configs)
    
    markdown_splits = lazy_markdown_splitter(document_stream)
    chunk_splits =  lazy_split_text_in_chunk(markdown_splits)

    embedding_model = MSMarcoEmbeddings(emb_model, token=settings.hf_token)

    vector_store = PGVector(
        embeddings=embedding_model,
        collection_name="developer_docs",
        connection=database_url,
        use_jsonb=True,
        create_extension=False,
    )

    total = 0
    async for batch in lazy_load_batches(chunk_splits, batch_size): 
        vector_store.aadd_documents(batch)
        total += len(batch)
        print(f"Inserted {total} chunks...")


@time_async
async def delete_all_embeddings(database_url, collection_name="developer_docs"):
    """Truncate all embeddings from the PGVector collection without dropping it."""

    engine =  create_async_engine(database_url)
    async with engine.begin() as conn:
        # Delete all rows in the embedding table belonging to this collection
        await conn.execute(text("""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection
                WHERE name = :collection_name
            )
        """), {"collection_name": collection_name})
    
    await engine.dispose()
    
    print(f"Cleared all embeddings from collection '{collection_name}'.")
    
            