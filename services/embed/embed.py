import os
import anyio
from embed_utils import lazy_embed_chunks_to_json
import logging
import sys

logger = logging.getLogger(__name__)

emb_model_name = os.environ.get('EMBED_MODEL_NAME', 'msmarco-bert-base-dot-v5')

# Repo configs, to extract .md files
repo_configs = {
    "repo": os.environ.get('REPO', 'pydantic'),
    "owner": os.environ.get('OWNER', 'pydantic'),
    "branch": os.environ.get('BRANCH', 'main'),
}


async def main():
    logger.info("Starting embedding process...")
    await lazy_embed_chunks_to_json(repo_configs, emb_model_name, fetch_batch_size=512)
    logger.info("Embedding process completed.")


if __name__ == "__main__":
    anyio.run(main)
