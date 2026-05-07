from logging_setup import setup_logging

setup_logging(log_file="logs/app.log")

from embed_utils import delete_all_embeddings

from config import settings
import anyio

emb_model_name = settings.emb_model_name
database_url = settings.database_url.get_secret_value()

# Repo configs, to extract .md files
repo_configs = {
    "repo": "pydantic",
    "owner": "pydantic",
    "branch": "main",
}


async def main():

    await lazy_insert_chunk_batches(
        repo_configs, emb_model_name, database_url, batch_size=512
    )
    # await delete_all_embeddings(settings.database_url.get_secret_value(), collection_name="developer_docs")


if __name__ == "__main__":
    anyio.run(main)
