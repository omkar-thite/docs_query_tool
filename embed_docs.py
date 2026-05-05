from logging_setup import setup_logging
setup_logging(log_file="logs/app.log")

from embed_utils import MSMarcoEmbeddings, lazy_insert_chunk_batches

from config import settings
import anyio
emb_model = 'msmarco-bert-base-dot-v5'

# Repo configs, to extract .md files
repo_configs = {'repo': 'pydantic',
               'owner': 'pydantic',
               'branch': 'main',
            #    'prefix': 'docs'
               }

async def main():

    await lazy_insert_chunk_batches(repo_configs, emb_model, settings.database_url.get_secret_value(), batch_size=32)


if __name__ == "__main__":
    anyio.run(main)
