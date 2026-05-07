from logging_setup import setup_logging
from config import settings
import anyio
from embed_utils import lazy_embed_chunks_to_json

setup_logging(log_file="logs/app.log")

emb_model_name = settings.emb_model_name
database_url = settings.database_url.get_secret_value()

# Repo configs, to extract .md files
repo_configs = {
    "repo": "pydantic",
    "owner": "pydantic",
    "branch": "main",
}


async def main():

    await lazy_embed_chunks_to_json(
        repo_configs, emb_model_name, database_url, batch_size=512
    )
    # await delete_all_embeddings(settings.database_url.get_secret_value(), collection_name="developer_docs")


if __name__ == "__main__":
    anyio.run(main)
