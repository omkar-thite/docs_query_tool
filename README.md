# RAG Pipeline for Documentations
[Under Development]  

This is supposed to be an offline tool for developers. The idea is to get a quick documentation reference for a given developer query.  
eg. a developer want to implement certain niche logic, but is not aware if library already gives training wheels. So he/she goes to library's documentation and searches for relevant section. This tool aims to automate this process, it take what developer wants as a query, search through doc's embeddings, and if something similar is found then returns it in thorugh LLM. Tool will be developed in a way so smaller models can be used in local machines as LLMs. Currently it calls google's gemini API to generate response. 

In short, a Retrieval-Augmented Generation (RAG) pipeline that extracts Markdown documentation from GitHub repositories *(currently configured for pydantic)*, processes text into hierarchical chunks, generates vector embeddings, and stores them in PostgreSQL via `pgvector`. Queries are answered by Google's Gemini LLM using context retrieved through vector similarity search.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Database | PostgreSQL + `pgvector` |
| Embeddings | `sentence-transformers` (`msmarco-bert-base-dot-v5`) |
| LLM | Google GenAI SDK (`gemini-2-flash-preview`) |
| Text Processing | `langchain-text-splitters`, `langchain-core` |
| DB Client | `psycopg`, `pgvector`, `sqlalchemy` |
| Config | `pydantic-settings` |
| Async I/O | `anyio`, `httpx`, `asyncio` |

---

## Project Structure

```
.
├── config.py               # Environment variables and DB config (includes Docker Compose reference)
├── embed_docs.py           # Entry point: fetches repo configs and triggers the embedding pipeline
├── embed_utils.py          # GitHub API client, file tree traversal, chunking, and GPU embedding
├── logging_setup.py        # App-wide logging with rotating file handlers and console output
├── migrate_to_pgvector.py  # DB schema init and bulk loading of parents.json / children.json
├── model_call.py           # Google GenAI API integration and RAG prompt construction
└── retrieval.py            # Embedding model cache and pgvector similarity search
```

---

## Installation

1. Clone the repository.

2. Install dependencies:
   ```bash
   pip install pydantic-settings httpx langchain-core langchain-text-splitters \
               sentence-transformers psycopg pgvector google-genai anyio tqdm sqlalchemy
   ```

3. Start the PostgreSQL database with the `pgvector` extension:
   ```bash
   docker-compose up -d
   ```
   > A reference `docker-compose.yml` snippet is provided inside `config.py`.

4. *(Optional)* Create a dedicated database user and database as described in the `.env` example below, and use those credentials for vector storage and retrieval.

5. Dev setup

   uv sync
   uv run pre-commit install

---

## Configuration

Create a `.env` file in the project root:

```env
# .env.example

# GitHub Authentication (required for fetching repository docs)
GITHUB_API_TOKEN=YOUR_GITHUB_TOKEN

# HuggingFace Token (required for downloading models)
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN

# Google Gemini API (required for LLM generation)
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

# Database Credentials (required by psycopg / migrate_to_pgvector)
DB_HOST=localhost
DB_PORT=5432
APP_USER=YOUR_POSTGRES_USER
APP_PASSWORD=YOUR_POSTGRES_PASSWORD
APP_DB=YOUR_POSTGRES_DB

# Docker Compose Variables
POSTGRES_USER=YOUR_POSTGRES_USER
POSTGRES_PASSWORD=YOUR_POSTGRES_PASSWORD
POSTGRES_DB=YOUR_POSTGRES_DB

# SQLAlchemy Connection URLs
DATABASE_URL=postgresql://user:password@host:port/dbname
ASYNC_DATABASE_URL=postgresql+asyncpg://user:password@host:port/dbname
```

---

## Usage

### 1. Extract and embed documentation

Fetches Markdown files from the configured GitHub repository, chunks the text hierarchically, computes embeddings, and saves the results to `children.json` and `parents.json`.

```bash
python embed_docs.py
```

### 2. Migrate embeddings to PostgreSQL

Loads the processed JSON files into the `pgvector` database, handling schema creation, HNSW index setup, and bulk insertion.

```bash
python migrate_to_pgvector.py
```

### 3. Query the pipeline

Embeds the query, retrieves the top-`k` similar child chunks, maps them to their parent context, and passes everything to Gemini for a grounded answer.

```bash
python model_call.py "How do I define a base settings model?"
```

---

## API Reference

### `retrieval.py` — Vector Search

**`class MSMarcoEmbeddings`**
Manages loading, caching, and inference for the `msmarco-bert-base-dot-v5` model.

| Method | Signature | Description |
|---|---|---|
| `embed_documents` | `(texts: List[str]) -> List[List[float]]` | Batch-encodes a list of documents |
| `embed_query` | `(text: str) -> List[float]` | Encodes a single query string |

**`get_similar_chunks(query: str, k: int = 3) -> dict`**
Connects to PostgreSQL and performs HNSW inner-product similarity search, returning the parent document content for the top-`k` matches.

---

### `embed_utils.py` — Document Processing

**`visit_main_tree_and_extract_docs_files(repo_configs)`**
Asynchronously traverses a GitHub repository tree and yields `Document` objects for all `.md` files found.

**`lazy_load_batches(file_documents_generator, batch_size=256)`**
Applies a hierarchical parent/child split to documents and yields chunks in batches for memory-efficient processing.

---

### `model_call.py` — LLM Integration

**`build_rag_prompt(query: str, chunks: list[str]) -> str`**
Constructs the prompt payload by injecting retrieved context chunks alongside the user query.

**`answer_query(query: str, model: str, k: int = 3)`**
Orchestrates the full RAG loop: retrieval → prompt construction → Gemini API call → response.

---

